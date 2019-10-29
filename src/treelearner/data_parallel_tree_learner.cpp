/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cstring>
#include <tuple>
#include <vector>

#ifdef __JETBRAINS_IDE__
#define USE_MPI
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif
#include "mxx/collective.hpp"

#include "parallel_tree_learner.h"

namespace LightGBM {

template <typename TREELEARNER_T>
DataParallelTreeLearner<TREELEARNER_T>::DataParallelTreeLearner(const Config* config)
  :TREELEARNER_T(config) {
}

template <typename TREELEARNER_T>
DataParallelTreeLearner<TREELEARNER_T>::~DataParallelTreeLearner() {
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::Init(const Dataset* train_data, bool is_constant_hessian) {
  // initialize SerialTreeLearner
  TREELEARNER_T::Init(train_data, is_constant_hessian);
  // Get local rank and global machine size
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();
  // allocate buffer for communication
  size_t buffer_size = this->train_data_->NumTotalBin() * sizeof(HistogramBinEntry);

//  Log::Info("%d: Buffer size at init: %d", rank_, buffer_size);

  input_buffer_.resize(buffer_size);
  output_buffer_.resize(buffer_size);

  is_feature_aggregated_.resize(this->num_features_);

  block_start_.resize(num_machines_);
  block_len_.resize(num_machines_);

  buffer_write_start_pos_.resize(this->num_features_);
  buffer_read_start_pos_.resize(this->num_features_);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::ResetConfig(const Config* config) {
  TREELEARNER_T::ResetConfig(config);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::BeforeTrain() {
  TREELEARNER_T::BeforeTrain();
  // generate feature partition for current tree
  std::vector<std::vector<int>> feature_distribution(num_machines_, std::vector<int>());
  std::vector<int> num_bins_distributed(num_machines_, 0);
  int default_bins = 0;
  for (int i = 0; i < this->train_data_->num_total_features(); ++i) {
    int inner_feature_index = this->train_data_->InnerFeatureIndex(i);
    if (inner_feature_index == -1) { continue; }
    if (this->is_feature_used_[inner_feature_index]) {
      // tvas: This tries to distribute the number of bins to each machine evenly
      //  by finding which machine has the least num of bins up till now, and using that
      //  to assign the bins for the next feature in the loop
      int cur_min_machine = static_cast<int>(ArrayArgs<int>::ArgMin(num_bins_distributed));
      // tvas: feature_distribution assigns different features to each worker. I think these determine how each worker
      //  is later assigned per-feature splits to find the best one globally?
      feature_distribution[cur_min_machine].push_back(inner_feature_index);
      auto num_bin = this->train_data_->FeatureNumBin(inner_feature_index);
      // tvas: What does it mean when the following condition is true?
      if (this->train_data_->FeatureBinMapper(inner_feature_index)->GetDefaultBin() == 0) {
        default_bins++;
        num_bin -= 1;
      }
      num_bins_distributed[cur_min_machine] += num_bin;
    }
    is_feature_aggregated_[inner_feature_index] = false;
  }
  // get local used feature
  for (auto fid : feature_distribution[rank_]) {
    // tvas: This determines if a particular feature is aggregated at this machine,
    //   according to values of `feature_distribution` for the current rank.
    //   Features that are _not_ aggregated locally, do not take part in this worker's best split computation
    is_feature_aggregated_[fid] = true;
  }



//  std::cout << rank_ << ": default_bins: " << default_bins << std::endl;

  // tvas: There is a discrepancy between the "total bins" and the values provided here.
  //  My guess is the default bins somehow affect this.
//  std::cout << rank_ << ": num_bins_distributed: ";
//  for (auto value : num_bins_distributed) {
//    std::cout << value << ", ";
//  }
//  std::cout << std::endl;
//  std::cout << rank_ << ": unused_features: ";
//  for (size_t i = 0; i < this->num_features_; ++i) {
//    if (!this->is_feature_used_[i]) {
//      std::cout << i << ", ";
//    }
//  }
//  std::cout << std::endl;

//  std::cout << rank_ << ": feature_distribution: " << std::endl;
//  int machine = 0;
//  for (auto const& vector : feature_distribution) {
//    std::cout << "\tm: " <<machine << " feature_distribution: ";
//    for (auto value : vector) {
//      std::cout << value << ", ";
//    }
//    std::cout << std::endl;
//    machine++;
//  }
//  std::cout << std::endl;

  // get block start and block len for reduce scatter
  reduce_scatter_size_ = 0;
  for (int i = 0; i < num_machines_; ++i) {
    block_len_[i] = 0;
    for (auto fid : feature_distribution[i]) {
      auto num_bin = this->train_data_->FeatureNumBin(fid);
      if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0) {
        num_bin -= 1;
      }
      block_len_[i] += num_bin * sizeof(HistogramBinEntry);
    }
    reduce_scatter_size_ += block_len_[i];
  }

  block_start_[0] = 0;
  for (int i = 1; i < num_machines_; ++i) {
    block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
  }

//  std::cout << rank_ << ": block_start_: ";
//  for (auto value : block_start_) {
//    std::cout << value << ", ";
//  }
//  std::cout << std::endl;
//
//
//  std::cout << rank_ << ": block_len_: ";
//  for (auto value : block_len_) {
//    std::cout << value << ", ";
//  }
//  std::cout << std::endl;


  // get buffer_write_start_pos_
  int bin_size = 0;
  for (int i = 0; i < num_machines_; ++i) {
    for (auto fid : feature_distribution[i]) {
      // tvas: The buffer write pos is determined by the order of the features
      //  across the workers. Each fid has a corresponding number of bins that it requires
      //  and the positions where we write its output is **non-contiguous**.
      //  So simply iterating over the different features and assigning values according to index
      //  will be wrong. We need to take this into consideration.
      buffer_write_start_pos_[fid] = bin_size;
      auto num_bin = this->train_data_->FeatureNumBin(fid);
      if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0) { // tvas: Investigate how this fucks things up
        num_bin -= 1;
      }
//      if (rank_ == 0) {
//        std::cout << "fid: " << fid << ", num_bin: " << num_bin <<std::endl;
//      }
      bin_size += num_bin * sizeof(HistogramBinEntry);
    }
  }

  // get buffer_read_start_pos_
  bin_size = 0;
  for (auto fid : feature_distribution[rank_]) {
    buffer_read_start_pos_[fid] = bin_size;
    auto num_bin = this->train_data_->FeatureNumBin(fid);
    if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0) {
      num_bin -= 1;
    }
    bin_size += num_bin * sizeof(HistogramBinEntry);
  }

  // sync global data sumup info
  std::tuple<data_size_t, double, double> data(this->smaller_leaf_splits_->num_data_in_leaf(),
                                               this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians());
  int size = sizeof(data);
  std::memcpy(input_buffer_.data(), &data, size);
  // global sumup reduce
  Network::Allreduce(input_buffer_.data(), size, sizeof(std::tuple<data_size_t, double, double>), output_buffer_.data(), [](const char *src, char *dst, int type_size, comm_size_t len) {
    comm_size_t used_size = 0;
    const std::tuple<data_size_t, double, double> *p1;
    std::tuple<data_size_t, double, double> *p2;
    while (used_size < len) {
      p1 = reinterpret_cast<const std::tuple<data_size_t, double, double> *>(src);
      p2 = reinterpret_cast<std::tuple<data_size_t, double, double> *>(dst);
      std::get<0>(*p2) = std::get<0>(*p2) + std::get<0>(*p1);
      std::get<1>(*p2) = std::get<1>(*p2) + std::get<1>(*p1);
      std::get<2>(*p2) = std::get<2>(*p2) + std::get<2>(*p1);
      src += type_size;
      dst += type_size;
      used_size += type_size;
    }
  });
  // copy back
  std::memcpy(reinterpret_cast<void*>(&data), output_buffer_.data(), size);
  // set global sumup info
  this->smaller_leaf_splits_->Init(std::get<1>(data), std::get<2>(data));
  // init global data count in leaf
  global_data_count_in_leaf_[0] = std::get<0>(data);

//  std::cout << rank_ << " buffer_write_start_pos_: ";
//  for (auto value: buffer_write_start_pos_) {
//    std::cout << value << ", ";
//  }
//  std::cout << std::endl;
//
//  std::cout << rank_ << " buffer_read_start_pos_: ";
//  for (auto value : buffer_read_start_pos_) {
//    std::cout << value << ", ";
//  }
//  std::cout << std::endl;

}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplits() {
  TREELEARNER_T::ConstructHistograms(this->is_feature_used_, true); // TODO: Non-zero hists can probably be annotated here
  // construct local histograms
  size_t total_bins = 0;
  int unused_features = 0;
  size_t total_non_zero_bins = 0;
  // TODO: Use and gather mapping instead of map-reduce style vector we have now
  // Mapping from feature id to non-zero bin indices
//  std::unordered_map<int, std::vector<int>> feature_to_nonzero_bins(this->num_features_);
  // Vector containing non-zero bin objects
  std::vector<HistogramBinEntry> non_zero_bins_vector;
  non_zero_bins_vector.reserve(reduce_scatter_size_ / sizeof(HistogramBinEntry)); // Overestimation

  // Map-reduce style vector of (feature_id, bin_idx) pairs. We use these to perform the reduction at the root process
  std::vector<std::array<size_t , 2>> non_zero_bins_for_feature;
  non_zero_bins_for_feature.reserve(reduce_scatter_size_ / sizeof(HistogramBinEntry)); // Underestimation
//  #pragma omp parallel for schedule(static) // TODO: Add back once bugs are figured out
  for (size_t feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    if ((!this->is_feature_used_.empty() && this->is_feature_used_[feature_index] == false)) {
      unused_features++;
      continue;
    }
    FeatureHistogram& hist_for_feature = this->smaller_leaf_histogram_array_[feature_index];
    const HistogramBinEntry *  bin_entry_ptr = hist_for_feature.RawData();
    int non_zero_bins = 0;
    for (size_t bin_index = 0; bin_index < hist_for_feature.SizeOfHistgram() / sizeof(HistogramBinEntry); ++bin_index) {
      total_bins++;
      if (bin_entry_ptr->sum_gradients != 0.0 || bin_entry_ptr->sum_hessians != 0.0) {
        // Keep track of indices of non-zero bins
        non_zero_bins_for_feature.emplace_back(std::array<size_t , 2>{feature_index, bin_index});
        // copy to buffer
        non_zero_bins_vector.emplace_back(*bin_entry_ptr);
        // TODO: Remove check eventually
        CHECK(non_zero_bins_vector[total_non_zero_bins].sum_gradients == bin_entry_ptr->sum_gradients);
        non_zero_bins++;
        total_non_zero_bins++;
      }
      // Advance pointer to next bin
      bin_entry_ptr++;
    }
//    // copy to buffer
//    std::memcpy(input_buffer_.data() + buffer_write_start_pos_[feature_index],
//                hist_for_feature.RawData(),
//                hist_for_feature.SizeOfHistgram());
  }
  non_zero_bins_vector.resize(total_non_zero_bins); // tvas: Unnecessary?

//  std::cout << rank_ << ": Number of non-zero bins: " << total_non_zero_bins << "/" << total_bins << std::endl;

//  if (rank_ == 0) {
//    int limit = reduce_scatter_size_ / sizeof(HistogramBinEntry);
//    std::cout << "Original data on rank 0: ";
//    auto input_ptr = reinterpret_cast<const HistogramBinEntry *>(input_buffer_.data());
//    for (int i = 0; i < limit; ++i) {
//      std::cout << (input_ptr + i)->sum_gradients << ", ";
//    }
//    std::cout << std::endl;
//
//    std::cout << "Non-zero data on rank 0: ";
//    for (const auto &element : non_zero_bins_vector) {
//      std::cout << element.sum_gradients << ", ";
//    }
//    std::cout << std::endl;
//  }
//
//  Log::Info("[%d] reduce_scatter_size/HistogramSize: %d", rank_, reduce_scatter_size_ / sizeof(HistogramBinEntry));

  // Gather byte size of non-zero bins from each process
  std::vector<size_t> non_zero_bytes_per_process = mxx::gather(total_non_zero_bins * sizeof(HistogramBinEntry), 0);

  std::vector<size_t> non_zero_bins_per_process(num_machines_);

  // Convert bin byte sizes to number of bins per process for the root
  if (rank_  == 0) {
    int idx = 0;
    std::generate(non_zero_bins_per_process.begin(), non_zero_bins_per_process.end(),
                  [non_zero_bytes_per_process, idx] () mutable {
      return non_zero_bytes_per_process[idx++] / sizeof(HistogramBinEntry);});
  }

  // TODO: Figure out how to do custom datatype for mxx
  // Gather non-zero bins contents
  std::vector<char> gathered_data = mxx::gatherv(reinterpret_cast<char*>(non_zero_bins_vector.data()),
      total_non_zero_bins * sizeof(HistogramBinEntry), 0);
  // TODO: Can prolly create a container for HistogramBinEntry that also includes the feature id and bin id to avoid second gather
  // Gather non-zero bins locations per feature id
  std::vector<std::array<size_t, 2>>
      gathered_indices = mxx::gatherv(non_zero_bins_for_feature.data(), total_non_zero_bins, non_zero_bins_per_process, 0);

  // copy of buffer
  std::vector<char> reduced_data(input_buffer_.size());

  // Perform reduction at the root
  if (rank_ == 0) {
//    std::cout << "Number of gathered indices: " << gathered_indices.size() << std::endl;
//    std::cout << "Num non-zero bins per process: ";
//    for (int i = 0; i < num_machines_; ++i) {
//      std::cout << i << ": " << non_zero_bins_per_process[i] << " ";
//    }
//    std::cout << std::endl;
//    std::cout << "Number of gathered bins: " << gathered_data.size() / sizeof(HistogramBinEntry) << std::endl;

    // Perform map-reduce style reduction, based on provided (feature_id, bin_idx) pairs
    // This reduction is fine because the reduce operation (addition) is commutative
    for (size_t i = 0; i < gathered_indices.size(); ++i) {
      // index_pair: (feature_id, inner_bin_idx)
      const std::array<size_t, 2> &index_pair = gathered_indices[i];

      // Get the end position we will be writing the data to. The goal is to be equivalent to output_buffer_
      char* output_bin = reduced_data.data() +
          buffer_write_start_pos_[index_pair[0]] + (index_pair[1] * sizeof(HistogramBinEntry));

      // Use the built-in function to perform the reduction, reducing a single pair of HistogramBinEntry elements
      LightGBM::HistogramBinEntry::SumReducer(
          &gathered_data[i * sizeof(HistogramBinEntry)], output_bin, sizeof(HistogramBinEntry), sizeof(HistogramBinEntry));
    }
  }

//  int limit = reduce_scatter_size_ / sizeof(HistogramBinEntry);
//  std::cout << "Rank: " << rank_ << ": Original data: ";
//  auto input_ptr = reinterpret_cast<const HistogramBinEntry *>(input_buffer_.data());
//  for (int i = 0; i < limit; ++i) {
//    std::cout << (input_ptr + i)->sum_gradients << ", ";
//  }
//  std::cout << std::endl;
//
//  std::vector<char> output_copy(output_buffer_.size());
//  if (rank_ == 0) {
//    std::cout << "Gathered and reduced data: ";
//    for (int i = 0; i < limit; ++i) {
//      const HistogramBinEntry &bin_data = reinterpret_cast<HistogramBinEntry&>(reduced_data[i * sizeof(HistogramBinEntry)]);
//      std::cout << bin_data.sum_gradients << ", ";
//    }
//    std::cout << std::endl;
//  }

  std::vector<size_t> block_len_copy(block_len_.begin(), block_len_.end());
  if (rank_ == 0) {
    mxx::scatterv(reduced_data.data(), block_len_copy, output_buffer_.data(), block_len_copy[rank_], 0);
  } else {
    mxx::scatterv_recv(output_buffer_.data(), block_len_copy[rank_], 0);
  }


//  std::cout << "Rank " << rank_ << " scattered data: ";
//  for (int i = 0; i < limit; ++i) {
//    const HistogramBinEntry &bin_data = reinterpret_cast<HistogramBinEntry&>(output_copy[i * sizeof(HistogramBinEntry)]);
//    std::cout << bin_data.sum_gradients << ", ";
//  }
//  std::cout << std::endl;


  this->FindBestSplitsFromHistograms(this->is_feature_used_, true);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplitsFromHistograms(const std::vector<int8_t>&, bool) {
  std::vector<SplitInfo> smaller_bests_per_thread(this->num_threads_, SplitInfo());
  std::vector<SplitInfo> larger_bests_per_thread(this->num_threads_, SplitInfo());
  std::vector<int8_t> smaller_node_used_features(this->num_features_, 1);
  std::vector<int8_t> larger_node_used_features(this->num_features_, 1);
  if (this->config_->feature_fraction_bynode < 1.0f) {
    smaller_node_used_features = this->GetUsedFeatures(false);
    larger_node_used_features = this->GetUsedFeatures(false);
  }
  OMP_INIT_EX();
#pragma omp parallel for schedule(static)
  for (size_t feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_aggregated_[feature_index]) continue;
    const int tid = omp_get_thread_num();
    const int real_feature_index = this->train_data_->RealFeatureIndex(feature_index);
    // restore global histograms from buffer
    this->smaller_leaf_histogram_array_[feature_index].FromMemory(
      output_buffer_.data() + buffer_read_start_pos_[feature_index]);

    this->train_data_->FixHistogram(feature_index,
                                    this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians(),
                                    GetGlobalDataCountInLeaf(this->smaller_leaf_splits_->LeafIndex()),
                                    this->smaller_leaf_histogram_array_[feature_index].RawData());
    SplitInfo smaller_split;
    // find best threshold for smaller child
    this->smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
      this->smaller_leaf_splits_->sum_gradients(),
      this->smaller_leaf_splits_->sum_hessians(),
      GetGlobalDataCountInLeaf(this->smaller_leaf_splits_->LeafIndex()),
      this->smaller_leaf_splits_->min_constraint(),
      this->smaller_leaf_splits_->max_constraint(),
      &smaller_split);
    smaller_split.feature = real_feature_index;
    if (smaller_split > smaller_bests_per_thread[tid] && smaller_node_used_features[feature_index]) {
      smaller_bests_per_thread[tid] = smaller_split;
    }

    // only root leaf
    if (this->larger_leaf_splits_ == nullptr || this->larger_leaf_splits_->LeafIndex() < 0) continue;

    // construct histgroms for large leaf, we init larger leaf as the parent, so we can just subtract the smaller leaf's histograms
    this->larger_leaf_histogram_array_[feature_index].Subtract(
      this->smaller_leaf_histogram_array_[feature_index]);
    SplitInfo larger_split;
    // find best threshold for larger child
    this->larger_leaf_histogram_array_[feature_index].FindBestThreshold(
      this->larger_leaf_splits_->sum_gradients(),
      this->larger_leaf_splits_->sum_hessians(),
      GetGlobalDataCountInLeaf(this->larger_leaf_splits_->LeafIndex()),
      this->larger_leaf_splits_->min_constraint(),
      this->larger_leaf_splits_->max_constraint(),
      &larger_split);
    larger_split.feature = real_feature_index;
    if (larger_split > larger_bests_per_thread[tid] && larger_node_used_features[feature_index]) {
      larger_bests_per_thread[tid] = larger_split;
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_bests_per_thread);
  int leaf = this->smaller_leaf_splits_->LeafIndex();
  this->best_split_per_leaf_[leaf] = smaller_bests_per_thread[smaller_best_idx];

  if (this->larger_leaf_splits_ != nullptr &&  this->larger_leaf_splits_->LeafIndex() >= 0) {
    leaf = this->larger_leaf_splits_->LeafIndex();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_bests_per_thread);
    this->best_split_per_leaf_[leaf] = larger_bests_per_thread[larger_best_idx];
  }

  SplitInfo smaller_best_split, larger_best_split;
  smaller_best_split = this->best_split_per_leaf_[this->smaller_leaf_splits_->LeafIndex()];
  // find local best split for larger leaf
  if (this->larger_leaf_splits_->LeafIndex() >= 0) {
    larger_best_split = this->best_split_per_leaf_[this->larger_leaf_splits_->LeafIndex()];
  }

  // sync global best info
  SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->config_->max_cat_threshold);

  // set best split
  this->best_split_per_leaf_[this->smaller_leaf_splits_->LeafIndex()] = smaller_best_split;
  if (this->larger_leaf_splits_->LeafIndex() >= 0) {
    this->best_split_per_leaf_[this->larger_leaf_splits_->LeafIndex()] = larger_best_split;
  }
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::Split(Tree* tree, int best_Leaf, int* left_leaf, int* right_leaf) {
  TREELEARNER_T::Split(tree, best_Leaf, left_leaf, right_leaf);
  const SplitInfo& best_split_info = this->best_split_per_leaf_[best_Leaf];
  // need update global number of data in leaf
  global_data_count_in_leaf_[*left_leaf] = best_split_info.left_count;
  global_data_count_in_leaf_[*right_leaf] = best_split_info.right_count;
}

// instantiate template classes, otherwise linker cannot find the code
template class DataParallelTreeLearner<GPUTreeLearner>;
template class DataParallelTreeLearner<SerialTreeLearner>;

}  // namespace LightGBM

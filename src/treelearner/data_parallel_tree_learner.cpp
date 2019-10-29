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

  Log::Info("%d: Buffer size at init: %d", rank_, buffer_size);

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



  std::cout << rank_ << ": default_bins: " << default_bins << std::endl;

  // tvas: There is a discrepancy between the "total bins" and the values provided here.
  //  My guess is the default bins somehow affect this.
  std::cout << rank_ << ": num_bins_distributed: ";
  for (auto value : num_bins_distributed) {
    std::cout << value << ", ";
  }
  std::cout << std::endl;
  std::cout << rank_ << ": unused_features: ";
  for (int i = 0; i < this->num_features_; ++i) {
    if (!this->is_feature_used_[i]) {
      std::cout << i << ", ";
    }
  }
  std::cout << std::endl;

  std::cout << rank_ << ": feature_distribution: " << std::endl;
  int machine = 0;
  for (auto const& vector : feature_distribution) {
    std::cout << "\tm: " <<machine << " feature_distribution: ";
    for (auto value : vector) {
      std::cout << value << ", ";
    }
    std::cout << std::endl;
    machine++;
  }
  std::cout << std::endl;

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

  std::cout << rank_ << ": block_start_: ";
  for (auto value : block_start_) {
    std::cout << value << ", ";
  }
  std::cout << std::endl;


  std::cout << rank_ << ": block_len_: ";
  for (auto value : block_len_) {
    std::cout << value << ", ";
  }
  std::cout << std::endl;


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
      if (rank_ == 0) {
        std::cout << "fid: " << fid << ", num_bin: " << num_bin <<std::endl;
      }
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

  std::cout << rank_ << " buffer_write_start_pos_: ";
  for (auto value: buffer_write_start_pos_) {
    std::cout << value << ", ";
  }
  std::cout << std::endl;

  std::cout << rank_ << " buffer_read_start_pos_: ";
  for (auto value : buffer_read_start_pos_) {
    std::cout << value << ", ";
  }
  std::cout << std::endl;

}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplits() {
  TREELEARNER_T::ConstructHistograms(this->is_feature_used_, true); // TODO: Non-zero hists can probably be annotated here
  // construct local histograms
  size_t total_bins = 0;
  int unused_features = 0;
  size_t total_non_zero_bins = 0;
  // Mapping from feature id to non-zero bin indices
  std::unordered_map<int, std::vector<int>> feature_to_nonzero_bins(this->num_features_);
  // Vector containing non-zero bin objects
  std::vector<HistogramBinEntry> non_zero_bins_vector;
  non_zero_bins_vector.reserve(reduce_scatter_size_ / sizeof(HistogramBinEntry)); // Overestimation

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
        if (rank_ == 0) {
          std::cout << "Original hist grad entry: " << bin_entry_ptr->sum_gradients << std::endl;
          std::cout << "Copied hist grad entry: " << non_zero_bins_vector[total_non_zero_bins].sum_gradients << std::endl;
        }
        CHECK(non_zero_bins_vector[total_non_zero_bins].sum_gradients == bin_entry_ptr->sum_gradients);
        non_zero_bins++;
        total_non_zero_bins++;
        if (rank_ == 0) {
          std::cout << "Non-zero data up to now, size:  " << non_zero_bins_vector.size() << " :";
          for (int i = 0; i < total_non_zero_bins; ++i) {
            const auto &element = non_zero_bins_vector[i];
            std::cout //<< "address: " << static_cast<void*>(non_zero_bins_vector.data() + i)
              << i << " : (grad: "<< element.sum_gradients << ", hess: " << element.sum_hessians << ", cnt: "
              << element.cnt << "), ";
          }
          std::cout << std::endl;
        }
      }
      // Advance pointer to next bin
      bin_entry_ptr++;
    }
    non_zero_bins_vector.resize(total_non_zero_bins);
    // copy to buffer
    std::memcpy(input_buffer_.data() + buffer_write_start_pos_[feature_index],
                hist_for_feature.RawData(),
                hist_for_feature.SizeOfHistgram());
  }

  std::cout << rank_ << ": Number of non-zero bins: " << total_non_zero_bins << "/" << total_bins << std::endl;

  if (rank_ == 0) {
    int limit = reduce_scatter_size_ / sizeof(HistogramBinEntry);
    std::cout << "Original data: ";
    auto input_ptr = reinterpret_cast<const HistogramBinEntry *>(input_buffer_.data());
    for (int i = 0; i < limit; ++i) {
      std::cout << (input_ptr + i)->sum_gradients << ", ";
    }
    std::cout << std::endl;

    std::cout << "Non-zero data: ";
    for (const auto &element : non_zero_bins_vector) {
      std::cout << element.sum_gradients << ", ";
    }
    std::cout << std::endl;
  }

  Log::Info("[%d] unused_features: %d", rank_, unused_features);
  Log::Info("[%d] reduce_scatter_size: %d", rank_, reduce_scatter_size_);
  Log::Info("[%d] reduce_scatter_size/HistogramSize: %d", rank_, reduce_scatter_size_ / sizeof(HistogramBinEntry));
  Log::Info("[%d] input_buffer char size: %d", rank_, input_buffer_.size());
  Log::Info("[%d] input_buffer size / Histogram size: %d", rank_, input_buffer_.size() / sizeof(HistogramBinEntry));

  // Gather number of non-zero bins from each process
  std::vector<size_t> non_zero_per_process = mxx::gather(total_non_zero_bins, 0);

  // Gather non-zero bins contents
  std::vector<HistogramBinEntry> gathered_data = mxx::gatherv(non_zero_bins_vector.data(), total_non_zero_bins, 0);
  // TODO: Can prolly create a container for HistogramBinEntry that also includes the feature id and bin id to avoid second gather
  // Gather non-zero bins locations per feature id
  std::vector<std::array<size_t, 2>>
      gathered_indices = mxx::gatherv(non_zero_bins_for_feature.data(), total_non_zero_bins, non_zero_per_process, 0);

  // Perform reduction at the root
  if (rank_ == 0) {
    for (size_t i = 0; i < gathered_indices.size(); ++i) {
      // index_pair: (feature_id, inner_bin_idx)
      const std::array<size_t, 2> &index_pair = gathered_indices[i];

      // copy to buffer
      std::vector<char> input_copy(input_buffer_.size());
      auto histogram_bin = input_copy.data() + buffer_write_start_pos_[index_pair[0]] + (index_pair[1] * sizeof(HistogramBinEntry));

      LightGBM::HistogramBinEntry::SumReducer(
          reinterpret_cast<char *>(&gathered_data[i]), histogram_bin, sizeof(HistogramBinEntry), sizeof(HistogramBinEntry));
    }
  }


  // TODO: Verify that the local reduction works as expected, then perform a scatter
  size_t total_zero_gradients = 0;
  size_t total_zero_hess = 0;

  if (rank_ == 0) {
    int limit = 119;
//    std::cout << "Original data: ";
    auto input_ptr = reinterpret_cast<const HistogramBinEntry *>(input_buffer_.data());
//    for (int i = 0; i < limit; ++i) {
//      std::cout << (input_ptr + i)->sum_gradients << ", ";
//    }
//    std::cout << std::endl;
//
//    std::cout << "Gathered data: ";
//    for (int i = 0; i < limit; ++i) {
//      std::cout << gathered_data[i].sum_gradients << ", ";
//    }
//    std::cout << std::endl;

    uint different_values = 0;
    for (size_t i = 0; i < total_bins; ++i) {
      if (gathered_data[i].sum_gradients != (input_ptr + i)->sum_gradients) {
        if (i < total_bins) { // Only care about the "local" gradients as sent (?) by the root process
          different_values++;
          Log::Warning("Found difference between gathered, %f, and input, %f, grad at %zu",
                       gathered_data[i].sum_gradients, (input_ptr + i)->sum_gradients, i);
        }
      }
      if (gathered_data[i].sum_gradients == 0.0) {
        total_zero_gradients++;
      }
      if (gathered_data[i].sum_hessians == 0.0) {
        total_zero_hess++;
      }
    }
    std::cout << std::endl;
    if (different_values > 0) {
      Log::Warning("Number of different values between gathered and input: %d", different_values);
    }
//    Log::Info("[%d] Total zero gradients: %d/%d", Network::rank(), total_zero_gradients, gathered_data.size());
//    Log::Info("[%d] Total zero hessians: %d/%d", Network::rank(), total_zero_hess, gathered_data.size());
  }

  // Reduce scatter for histogram
  // TODO: Here is where the hist communication happens. They use raw byte arrays and then do casts to
  //  HistogramBinEntry inside the reduce function. The byte array gets piecemeal populated in the parallel
  //  loop above (where is it inited/allocated though?), in parallel, over features (do different workers have
  //  same number of features?. The idea here is to not populate when a histogram is empty
  //  and have gather-gatherv-local reduce step. At first we can skip (grad==0, hess==0) hists, and
  //  do a raw float aggregation as a second iteration.
  Log::Info("[%d] input_buffer elements: %d", rank_, input_buffer_.size() / sizeof(HistogramBinEntry));
  Log::Info("[%d] output_buffer elements: %d", rank_, output_buffer_.size() / sizeof(HistogramBinEntry));
  Log::Info("\n");
  Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, sizeof(HistogramBinEntry), block_start_.data(),
                         block_len_.data(), output_buffer_.data(), static_cast<comm_size_t>(output_buffer_.size()), &HistogramBinEntry::SumReducer);
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
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
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

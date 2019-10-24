/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cstring>
#include <tuple>
#include <vector>

#ifdef USE_MPI
#include <mpi.h>
#include "mxx/collective.hpp"
#endif

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
  for (int i = 0; i < this->train_data_->num_total_features(); ++i) {
    int inner_feature_index = this->train_data_->InnerFeatureIndex(i);
    if (inner_feature_index == -1) { continue; }
    if (this->is_feature_used_[inner_feature_index]) {
      int cur_min_machine = static_cast<int>(ArrayArgs<int>::ArgMin(num_bins_distributed));
      feature_distribution[cur_min_machine].push_back(inner_feature_index);
      auto num_bin = this->train_data_->FeatureNumBin(inner_feature_index);
      if (this->train_data_->FeatureBinMapper(inner_feature_index)->GetDefaultBin() == 0) {
        num_bin -= 1;
      }
      num_bins_distributed[cur_min_machine] += num_bin;
    }
    is_feature_aggregated_[inner_feature_index] = false;
  }
  // get local used feature
  for (auto fid : feature_distribution[rank_]) {
    is_feature_aggregated_[fid] = true;
  }

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

  // get buffer_write_start_pos_
  int bin_size = 0;
  for (int i = 0; i < num_machines_; ++i) {
    for (auto fid : feature_distribution[i]) {
      buffer_write_start_pos_[fid] = bin_size;
      auto num_bin = this->train_data_->FeatureNumBin(fid);
      if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0) { // tvas: Investigate how this fucks things up
        num_bin -= 1;
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
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplits() {
  TREELEARNER_T::ConstructHistograms(this->is_feature_used_, true);
  // construct local histograms
//  #pragma omp parallel for schedule(static)
  size_t total_histograms = 0;
  size_t zero_gradients = 0;
  size_t zero_hess = 0;

  int count = 2;
  MPI_Aint displacements[3] = {offsetof(HistogramBinEntry, sum_gradients), offsetof(HistogramBinEntry, sum_hessians),
                               offsetof(HistogramBinEntry, cnt)};
  const int block_lengths[2] = {2, 1};
  MPI_Datatype array_of_types[2] = {MPI_DOUBLE, MPI_INT};

  MPI_Datatype histogram_type;
//  MPI_Datatype tmp_type;
//  MPI_Aint lb, extent;

  MPI_Type_create_struct( count, block_lengths, displacements,
                          array_of_types, &histogram_type);
//  MPI_Type_get_extent(tmp_type, &lb, &extent); // tvas: Necessary?
//  MPI_Type_create_resized( tmp_type, lb, extent, &histogram_type);
  MPI_Type_commit(&histogram_type);

  int unused_features = 0;
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index) {
    if ((!this->is_feature_used_.empty() && this->is_feature_used_[feature_index] == false)) {
      unused_features++;
      continue;
    }

    // tvas: Count the zero gradients and hessians
    FeatureHistogram& hist_for_feature = this->smaller_leaf_histogram_array_[feature_index];
    HistogramBinEntry const*  bin_entry_ptr = hist_for_feature.RawData();
    for (size_t i = 0; i < hist_for_feature.SizeOfHistgram() / sizeof(HistogramBinEntry); ++i) {
      total_histograms++;
      if (bin_entry_ptr->sum_gradients == 0.0) {
        zero_gradients++;
      }
      if (bin_entry_ptr->sum_hessians == 0.0) {
        zero_hess++;
      }
      // Advance pointer to next histogram
      bin_entry_ptr++;
    }
    // copy to buffer
    std::memcpy(input_buffer_.data() + buffer_write_start_pos_[feature_index],
                this->smaller_leaf_histogram_array_[feature_index].RawData(),
                this->smaller_leaf_histogram_array_[feature_index].SizeOfHistgram());
  }

  Log::Info("[%d] unused_features: %d", rank_, unused_features);
  Log::Info("[%d] reduce_scatter_size: %d", rank_, reduce_scatter_size_);
  Log::Info("[%d] reduce_scatter_size/HistogramSize: %d", rank_, reduce_scatter_size_ / sizeof(HistogramBinEntry));
  Log::Info("[%d] input_buffer char size: %d", rank_, input_buffer_.size());
  Log::Info("[%d] input_buffer size / Histogram size: %d", rank_, input_buffer_.size() / sizeof(HistogramBinEntry));

  Log::Info("[%d] Zero gradients: %d/%d", Network::rank(), zero_gradients, total_histograms);
  Log::Info("[%d] Zero hessians: %d/%d", Network::rank(), zero_hess, total_histograms);

  std::vector<HistogramBinEntry> gathered_data;
  if (rank_ == 0) {
    gathered_data.resize(total_histograms * num_machines_);
  }

  mxx::gather(input_buffer_.data(), static_cast<int>(reduce_scatter_size_),
      reinterpret_cast<char *>((gathered_data.data())), 0);

//  MPI_Gather(input_buffer_.data(), total_histograms, histogram_type,
//             gathered_data.data(), total_histograms, histogram_type, 0, MPI_COMM_WORLD);

  size_t total_zero_gradients = 0;
  size_t total_zero_hess = 0;

  if (rank_ == 0) {
    int limit = 119;
    std::cout << "Original data: ";
    auto input_ptr = reinterpret_cast<const HistogramBinEntry *>(input_buffer_.data());
    for (int i = 0; i < limit; ++i) {
      std::cout << (input_ptr + i)->sum_gradients << ", ";
    }
    std::cout << std::endl;

    std::cout << "Gathered data: ";
    for (int i = 0; i < limit; ++i) {
      std::cout << gathered_data[i].sum_gradients << ", ";
    }
    std::cout << std::endl;

    int different_values = 0;
    for (size_t i = 0; i < total_histograms * num_machines_; ++i) {
      if (gathered_data[i].sum_gradients != (input_ptr + i)->sum_gradients) {
        if (i < total_histograms) { // Only care about the "local" gradients as sent (?) by the root process
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
    Log::Warning("Number of different values between gathered and input: %d", different_values);
    Log::Info("[%d] Total zero gradients: %d/%d", Network::rank(), total_zero_gradients, gathered_data.size());
    Log::Info("[%d] Total zero hessians: %d/%d", Network::rank(), total_zero_hess, gathered_data.size());
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

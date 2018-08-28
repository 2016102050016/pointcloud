#include"Forest.h"


RandomForest::RandomForest(int numTrees, int maxDepth, int minSamplesPerLeaf, float giniThresh) :
	_numTrees(numTrees),
	_maxDepth(maxDepth),
	_minSamplesPerLeaf(minSamplesPerLeaf),
	_giniThresh(giniThresh),
	_trainSample(nullptr),
	_forest(_numTrees, nullptr)
{
	std::cout << "The number of trees in the forest is: " << _numTrees << std::endl;
	std::cout << "The max depth of a single tree is: " << _maxDepth << std::endl;
	std::cout << "The minimal number of samples at a leaf node is: " << _minSamplesPerLeaf << std::endl;
	std::cout << "The gini threshold for splitting the samples is: " << _giniThresh << std::endl;
}

RandomForest::~RandomForest()
{
	if (_trainSample != nullptr)
	{
		delete _trainSample;
		_trainSample = nullptr;
	}
}

void RandomForest::train(Eigen::MatrixXf *trainset, Eigen::VectorXi *labels, Eigen::MatrixXi *indices,
						 Eigen::MatrixXf *dists, int numClasses, int numFeatsPerNode)
{
	if (_numTrees < 1)
	{
		std::cout << "Total number of trees must be bigger than 0." << std::endl;
		std::cerr << "Training not started." << std::endl;
		return;
	}
	if (_maxDepth < 1)
	{
		std::cout << "The max depth must be bigger than 0." << std::endl;
		std::cerr << "Training not started." << std::endl;
		return;
	}
	if (_minSamplesPerLeaf < 2)
	{
		std::cout << "The minimal number of samples at a leaf node must be greater than 1." << std::endl;
		std::cerr << "Training not started." << std::endl;
		return;
	}

	int _numSamples = trainset->rows();
	_numClasses = numClasses;
	_numFeatsPerNode = numFeatsPerNode;
	int _numSelectedSamples = _numSamples * 0.7;

	// initializing the trees
	for (int i = 0; i < _numTrees; ++i)
	{
		_forest[i] = new Tree(_maxDepth, _numFeatsPerNode, _minSamplesPerLeaf, _giniThresh);
	}

	// this object holds the whole training dataset
	_trainSample = new Sample(trainset, labels, indices, dists, _numClasses, _numFeatsPerNode);

	// selected samples
	Eigen::VectorXi selectedSamplesId(_numSelectedSamples);
	// tree training starts
	for (int i = 0; i < _numTrees; ++i)
	{
		std::cout << "Training tree No. " << i << std::endl;

		// randomly sample 2/3 of the points with replacement from training set
		Sample *sample = new Sample(_trainSample);
		sample->randomSampleDataset(selectedSamplesId, _numSelectedSamples);

		_forest[i]->train(sample);
		delete sample;
	}
}

//void RandomForest::predict(Eigen::MatrixXf &dataset, int &res)
//{
//	// prediction from every tree
//	std::vector<float> results(_numClasses, 0);
//	for (int i = 0; i < _numTrees; ++i)
//	{
//		Result r;
//		r.label = 0;
//		r.prob = 0;
//		r = _forest[i]->predict(dataset);
//		results[r.label] += r.prob;
//	}
//	int maxProbLabel = 0;
//	float maxProb = results[0];
//	for (int i = 1; i < _numClasses; ++i)
//	{
//		if (results[i] > maxProb)
//		{
//			maxProbLabel = i;
//			maxProb = results[i];
//		}
//	}
//	res = maxProbLabel;
//}
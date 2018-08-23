#include"Forest.h"

RandomForest::RandomForest(int numTrees, int maxDepth, int minSamplesPerLeaf, float giniThresh):
	_numTrees(numTrees),
	_maxDepth(maxDepth),
	_minSamplesPerLeaf(minSamplesPerLeaf),
	_giniThresh(giniThresh),
	_trainSample(nullptr),
	_forest(_numTrees, nullptr)
{}

RandomForest::~RandomForest()
{
	if (_trainSample != nullptr)
	{
		delete _trainSample;
		_trainSample = nullptr;
	}
}

void RandomForest::train(Eigen::MatrixXf &trainset, Eigen::VectorXi &labels, Eigen::MatrixXi &indices,
					Eigen::MatrixXf &dists, int numClasses, int numFeatsPerNode)
{
	int _numSamples = trainset.rows();
	_numClasses = numClasses;
	_numFeatsPerNode = numFeatsPerNode;
	
	// initializing the trees
	for (int i = 0; i < _numTrees; ++i)
	{
		_forest[i] = new Tree(_maxDepth, _numFeatsPerNode, _minSamplesPerLeaf, _giniThresh);
	}

	// this object holds the whole training dataset
	_trainSample =  new Sample(trainset, labels, indices, dists, _numClasses, _numFeatsPerNode);

	// set a seed for generating random numbers
	srand(time(NULL));

	// tree training starts
	for (int i = 0; i < _numTrees; ++i)
	{
		std::cout << "Training tree No. " << i << std::endl;
		// randomly sample 2/3 of the points with replacement from training set
		Sample *sample = new Sample(_trainSample);
		sample->randomSampleDataset(0.67);
		//Eigen::VectorXi samplesId = sample->getSelectedSamplesId();
		_forest[i]->train(sample);
		delete sample;
	}
}

void RandomForest::predict(Eigen::MatrixXf &dataset, int &res)
{
	// prediction from every tree
	std::vector<float> results(_numClasses, 0);
	for (int i = 0; i < _numTrees; ++i)
	{
		Result r;
		r.label = 0;
		r.prob = 0;
		r = _forest[i]->predict(dataset);
		results[r.label] += r.prob;
	}
	int maxProbLabel = 0;
	float maxProb = results[0];
	for (int i = 1; i < _numClasses; ++i)
	{
		if (results[i] > maxProb)
		{
			maxProbLabel = i;
			maxProb = results[i];
		}
	}
	res = maxProbLabel;
}
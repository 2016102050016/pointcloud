/**************************************************
This class is used for drawing samples and features
(test node functions) from the dataset. 
***************************************************/

#pragma once
#include "C:/Eigen/Eigen/Dense"
#include <algorithm>
#include <vector>
#include <random>

struct Features
{
	int _point1;
	int _point2;
	int _featType;
};

class Sample {
public:

	Sample(const Eigen::MatrixXf &dataset, const Eigen::VectorXi &labels, 
		   Eigen::MatrixXi &indexMat, Eigen::MatrixXf &distMat, int numClass, int numFeature);
	Sample(Sample* samples);
	Sample(const Sample* sample, Eigen::VectorXi &samplesId);

	// randomly select samples from dataset with replacement (bagging)
	// number of selected samples = percentage * total number of samples
	// return the indices of selected samples in vector form
	void randomSampleDataset(float percentage);

	// randomly sample features from each neighborhoood
	// given a neighborhood consisting of k points, the number of possible features
	// are k*(k-1)*n, where n is the projection operations, but only numSelectedFeatures
	// are randomly chosen from all these features
	void randomSampleFeatures();

	// return a matrix representing the neighborhood of the pointId-th point
	// whose shape is (k, d), where k is the number of nearest neighbors
	// and d is the dimention of each datapoint
	Eigen::MatrixXf buildNeighborhood(int pointId) const;

	// Compute the number of different classes in a Sample obejct
	inline int getNumClasses() { return _numClass; }

	// get the selected sample indices
	inline Eigen::VectorXi getSelectedSamplesId() { return _selectedSamplesId; }
	inline Eigen::VectorXi getSelectedSamplesId() const { return _selectedSamplesId; }

	// get the number of selected samples
	inline int getNumSelectedSamples() { return _numSelectedSamples; }

	// get the selected features
	inline std::vector<Features> getSelectedFeatures() { return _features; }

	// get the number of features sampled at each node
	inline int getNumFeatures() { return _numFeature; }
	inline int getNeighborhoodSize() { return _indexMat.cols(); }

	Eigen::VectorXi& _labels;
	Eigen::MatrixXf& _dataset;

private:

	// _indexMat stores the indices of nearest neighbors for each datapoint
	Eigen::MatrixXi& _indexMat;
	Eigen::MatrixXf& _distMat;

	// stores the indices of selected datapoints
	Eigen::VectorXi _selectedSamplesId;
	std::vector<Features> _features;
	int _numClass;
	int _numSelectedSamples;
	int _numFeature;

};

class Random
{
public:
	Random(int popSize, int sampleSize):
		_popSize(popSize),
		_sampleSize(sampleSize)
	{}

	std::vector<int> sampleWithoutReplacement()
	{
		std::vector<int> population;
		candidates(population);

		std::random_device rd;
		std::mt19937 gen(rd());
		std::shuffle(population.begin(), population.end(), gen);
		std::vector<int> samples(population.begin(), population.begin() + _sampleSize);
		return samples;
	}

private:
	void candidates(std::vector<int> &nums)
	{
		for (int i = 0; i < _popSize; ++i)
			nums.push_back(i);
	}
	int _popSize;
	int _sampleSize;
};
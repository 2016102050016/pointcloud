//#include "c:\Eigen\Eigen\Dense"
#include <iostream>
#include "InOut.h"
#include "Sample.h"
#include "FeatureFactory.h"
#include "Node.h"
#include "Forest.h"

int main(int argc, char** argv)
{
	// prepare dataset
	// instantiate a training object
	InOut trnObj;
	Eigen::MatrixXf trnData;
	Eigen::VectorXi trnLabels;
	
	trnObj.readPoints("./dataset/downsampled.txt", trnData);
	trnObj.readLabels("./dataset/downsampled.labels", trnLabels);
	
	// search for the k nearest neighbors for each point in the dataset
	// and store the indices and dists in two matrices for later use by
	// indexing instead of searching again
	Eigen::MatrixXi trnIndices;
	Eigen::MatrixXf trnDists;
	const int numNeighbors = 11;
	trnObj.searchNN(trnData, numNeighbors, trnIndices, trnDists);
	//std::cout << "indices\n" << trnIndices << std::endl;

	// instantiate a sampling object which is responsible for
	// sampling the dataset (bagging) and doing the comparison 
	// in each node
	const int numClasses = 8;
	const int numFeatures = 10;
	Sample smpObj(trnData, trnLabels, trnIndices, trnDists, numClasses, numFeatures);
	
	// ratio controls the number of samples to be selected from the dataset
	// in a sampling object and this sampling is one with replacement
	// (bagging process), usually ratio is set as 0.67 (two thirds)
	const float ratio = 0.1;
	smpObj.randomSampleDataset(ratio);
	//std::cout << smpObj.getSelectedSamplesId() << std::endl;
	/*Eigen::MatrixXf neighborhood;
	int pointId = 0;
	neighborhood = smpObj.buildNeighborhood(pointId);
	std::cout << "Neighborhood is: \n" << neighborhood << std::endl;
	std::vector<Features> features;
	smpObj.randomSampleFeatures(neighborhood);
	features = smpObj.getSelectedFeatures();
	for (auto f : features)
	{
		std::cout << f._point1 << " " << f._point2 << " " << f._featType;
		std::cout << std::endl;
	}*/
	//std::cout << neighborhood << std::endl;
	/*Random rd(100, 2);
	std::vector<int> samples = rd.sampleWithouReplacement();
	for (auto ele : samples)
		std::cout << ele << " ";
	std::cout << std::endl;*/
	//Eigen::VectorXi samplesId = smpObj.getSelectedSamplesId();
	//std::cout << "sample indices: \n" << samplesId;
	std::vector<Node*> nodes;
	Node* node = new Node();
	node->_samples = &smpObj;
	/*node->computeNodeGini();
	std::cout << "\nGini at this node is: " << node->getNodeGini() << std::endl;*/
	node->splitNodeByGini(nodes, 0, 10);

	RandomForest rf(10, 10, 10, 0);
	rf.train(trnData, trnLabels, trnIndices, trnDists, 8, 10);
	
	system("pause");
	return 0;
}
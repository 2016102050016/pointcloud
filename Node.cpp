#include "Node.h"
#include <vector>
#include "FeatureFactory.h"
#include <iostream>

Node::Node()
{
	_isLeaf = false;
	_samples = nullptr;
	_class = -1;
	_prob = 0.0;
	
}


float Node::computeGini(Eigen::VectorXi& samplesId)
{
	int numSamples = samplesId.size();
	int numClasses = _samples->getNumClasses();
	std::vector<int> probs(numClasses, 0);
	for (int i = 0; i < numSamples; ++i)
	{
		/*std::cout << "\nsample Id " << i << " is: " << samplesId[i];
		std::cout << _samples->_labels[12] << std::endl;*/
		probs[_samples->_labels[samplesId[i]]]++;
	}
	for (int i = 0; i < numClasses; ++i)
	{
		_probs.push_back(probs[i]);
	}
	float gini = 0;
	for (int i = 0; i < numClasses; ++i)
	{
		float p = probs[i] / (float)numClasses;
		gini += (p*p);
	}
	
	return 1 - gini;
}
// calculate the gini of the samples in this node
void Node::computeNodeGini()
{
	Eigen::VectorXi samplesId = _samples->getSelectedSamplesId();
	_gini = computeGini(samplesId);
}


//void Node::computeInfoGain(int id, float minInfoGain)
//{
//	Eigen::MatrixXf data = _samples->_dataset;
//	Eigen::VectorXi labels = _samples->_labels;
//	// randomly samples some points from the cloud
//	// and store the selected points id in sampleId
//	Eigen::VectorXi sampleId = _samples->getSelectedSamplesId();
//	int numSamples = sampleId.size();
//	// for this work, numClasses is 8
//	int numClasses = _samples->getNumClasses();
//	std::vector<Features> selectedFeatures = _samples->getSelectedFeatures();
//	int numFeats = _samples->getNumFeatures();
//
//	// the following variables are for storing the relevant information
//	// when the information gain is maximal
//	float maxInfoGain = 0;
//	Features maxFeatId;
//	float maxGiniLeft = 0;
//	float maxGiniRight = 0;
//	//float maxThresh = 0;
//	int maxSamplesLeft = 0;
//	std::vector<float> maxProbsLeft(numClasses, 0);
//	std::vector<float> maxProbsRight(numClasses, 0);
//	std::vector<int> maxIndicesInLeftChildNode;
//	std::vector<int> maxIndicesInRightChildNode;
//
//	// variables hold the params in the first loop
//	float fMaxInfoGain = 0;
//	Features fMaxFeatId;
//	float fMaxGiniLeft = 0;
//	float fMaxGiniRight = 0;
//	//float fMaxThresh = 0;
//	int fMaxSamplesLeft = 0;
//	std::vector<float> fMaxProbsLeft(numClasses, 0);
//	std::vector<float> fMaxProbsRight(numClasses, 0);
//	std::vector<int> fMaxIndicesInLeftChildNode;
//	std::vector<int> fMaxIndicesInRightChildNode;
//
//	// variables hold the params in the inner loops
//	float giniLeft = 0;
//	float giniRight = 0;
//	float infoGain = 0;
//	std::vector<float> probsLeft(numClasses, 0);
//	std::vector<float> probsRight(numClasses, 0);
//	std::vector<int> indicesInLeftChildNode;
//	std::vector<int> indicesInRightChildNode;
//
//	for (int i = 0; i < numFeats; ++i)
//	{
//		// apply each of the selected features on the selected samples
//		fMaxInfoGain = 0;
//		fMaxGiniLeft = 0;
//		fMaxGiniRight = 0;
//		// fMaxThresh = 0;
//		fMaxSamplesLeft = 0;
//		fMaxFeatId = selectedFeatures[0];
//		probsRight = _probs;
//		for (int j = 0; j < numSamples; ++j)
//		{
//			// apply a feature on each of the selected samples
//			giniLeft = 0;
//			giniRight = 0;
//			infoGain = 0;
//			// put this sample to left or right childe node based on the 
//			// result of the test function
//			Eigen::MatrixXf neighborhood = _samples->buildNeighborhood(sampleId[j]);
//			FeatureFactory nodeFeat(neighborhood, selectedFeatures[i]._featType, selectedFeatures[i]._point1, selectedFeatures[i]._point2);
//			
//			if (nodeFeat.computeFeature() == false)
//			{
//				probsLeft[labels[sampleId[j]]]++;
//				indicesInLeftChildNode.push_back(sampleId[j]);
//			}
//			else
//			{
//				probsRight[labels[sampleId[j]]]++;
//				indicesInRightChildNode.push_back(sampleId[j]);
//			}
//
//			for (int k = 0; k < numClasses; ++k)
//			{
//				float p = probsLeft[k] / (j + 1);
//				giniLeft += p * p;
//			}
//			giniLeft = 1 - giniLeft;
//			for (int k = 0; k < numClasses; ++k)
//			{
//				float p = probsRight[k] / (j + 1);
//				giniRight += p * p;
//			}
//			giniRight = 1 - giniRight;
//			float leftRatio = (j + 1.0) / numSamples;
//			float rightRatio = (numSamples - j - 1.0) / numSamples;
//			infoGain = _gini - leftRatio * giniLeft - rightRatio * giniRight;
//
//			if (infoGain > fMaxInfoGain)
//			{
//				fMaxGiniLeft = giniLeft;
//				fMaxGiniRight = giniRight;
//				fMaxInfoGain = infoGain;
//				fMaxProbsLeft = probsLeft;
//				fMaxProbsRight = probsRight;
//				fMaxIndicesInLeftChildNode = indicesInLeftChildNode;
//				fMaxIndicesInRightChildNode = indicesInRightChildNode;
//			}
//		}
//		if (fMaxInfoGain > maxInfoGain)
//		{
//			maxGiniLeft = fMaxGiniLeft;
//			maxGiniRight = fMaxGiniRight;
//			maxInfoGain = fMaxInfoGain;
//			maxProbsLeft = fMaxProbsLeft;
//			maxProbsRight =  fMaxProbsRight;
//			maxIndicesInLeftChildNode = fMaxIndicesInLeftChildNode;
//			maxIndicesInRightChildNode = fMaxIndicesInRightChildNode;
//		}
//	}
//	if (maxInfoGain < minInfoGain)
//		createLeaf();
//	else
//		_featId = maxFeatId;
//}


// split the node into two children and compute the gini in each child node
void Node::splitNodeByGini(std::vector<Node*> &nodes, int nodeId, float threshGini)
{
	Eigen::MatrixXf data = _samples->_dataset;
	Eigen::VectorXi labels = _samples->_labels;
	// randomly samples some points from the cloud
	// and store the selected points id in sampleId
	Eigen::VectorXi sampleId = _samples->getSelectedSamplesId();
	int numSamples = sampleId.size();
	// for this work, numClasses is 8
	int numClasses = _samples->getNumClasses();
	_samples->randomSampleFeatures();
	std::vector<Features> selectedFeatures = _samples->getSelectedFeatures();
	int numFeats = _samples->getNumFeatures();

	float bestGini = 0;
	float bestLeftGini = 0;
	float bestRightGini = 0;
	Features bestFeat = selectedFeatures[0];
	Eigen::VectorXi bestLeftChild;
	Eigen::VectorXi bestRightChild;

	for (int i = 0; i < numFeats; ++i)
	{
		// apply each of the selected features on the selected samples
		float gini = 0;
		float leftGini = 0;
		float rightGini = 0;
		std::vector<int> sampleIdLeft;
		std::vector<int> sampleIdRight;
		for (int j = 0; j < numSamples; ++j)
		{
			// put this sample to left or right childe node based on the 
			// result of the test function
			Eigen::MatrixXf neighborhood = _samples->buildNeighborhood(sampleId[j]);
			FeatureFactory nodeFeat(neighborhood, selectedFeatures[i]._featType, selectedFeatures[i]._point1, selectedFeatures[i]._point2);
			if (nodeFeat.computeFeature() == false)
				sampleIdLeft.push_back(sampleId[j]);
			else
				sampleIdRight.push_back(sampleId[j]);
		}
		// convert std::vector to Eigen::VectorXi
		int* ptr1 = &sampleIdLeft[0];
		int* ptr2 = &sampleIdRight[0];
		Eigen::VectorXi leftChild = Eigen::Map<Eigen::VectorXi>(ptr1, sampleIdLeft.size());
		Eigen::VectorXi rightChild = Eigen::Map<Eigen::VectorXi>(ptr2, sampleIdRight.size());
		/*Eigen::VectorXi leftChild(sampleIdLeft.data());
		Eigen::VectorXi rightChild(sampleIdRight.data());*/
		leftGini = computeGini(leftChild);
		rightGini = computeGini(rightChild);
		// gini of this node on feature[i]
		gini = leftChild.size() / (float) numSamples * leftGini + rightChild.size() / (float)numSamples * rightGini;
		if (i == 0)
		{
			// set baseline parameters using the first split result
			bestGini = gini;
			bestLeftGini = leftGini;
			bestRightGini = rightGini;
			bestFeat = selectedFeatures[i];
			bestLeftChild = leftChild;
			bestRightChild = rightChild;
		}
		else if (i!=0 and gini < bestGini)
		{
			// for all the gini on each feature, choose the feature
			// that leads to the lowest gini
			bestGini = gini;
			bestLeftGini = leftGini;
			bestRightGini = rightGini;
			bestFeat = selectedFeatures[i];
			bestLeftChild = leftChild;
			bestRightChild = rightChild;
		}
	}

	// TODO: add more stopping criteria
	if (bestGini < threshGini)
	{
		createLeaf();
	}
	else
	{
		_bestFeat = bestFeat;
		nodes[nodeId * 2 + 1] = new Node();
		nodes[nodeId * 2 + 2] = new Node();
		(nodes[nodeId * 2 + 1])->_gini = bestLeftGini;
		(nodes[nodeId * 2 + 1])->_gini = bestRightGini;
		Sample *leftSamples = new Sample(_samples, bestLeftChild);
		Sample *rightSamples = new Sample(_samples, bestRightChild);
		(nodes[nodeId * 2 + 1])->_samples = leftSamples;
		(nodes[nodeId * 2 + 1])->_samples = rightSamples;
	}

	
}

// make the most frequent class as the class of this leaf node
void Node::createLeaf()
{
	_class = 0;
	_prob = _probs[0];
	for (int i = 0; i < _samples->getNumClasses(); ++i)
	{
		if (_probs[i] > _prob)
		{
			_class = i;
			_prob = _probs[i];
		}
	}
	_prob /= _samples->getNumSelectedSamples();
	_isLeaf = true;
}

int Node::predict(Eigen::VectorXi &cloud, int pointId)
{
	return 0;
}

void Node::getResult(Result & res)
{
}

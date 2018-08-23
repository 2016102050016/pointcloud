#include "Tree.h"

Tree::Tree(int maxDepth, int numFeatPerNode, int minNumSamplesPerLeaf, float giniThresh):
	_maxDepth(maxDepth),
	_numFeatPerNode(numFeatPerNode),
	_minNumSamplesPerLeaf(minNumSamplesPerLeaf),
	_giniThresh(giniThresh),
	_numNodes(static_cast<int>(std::pow(2, _maxDepth)-1)),
	_treeNodes(_numNodes, nullptr)
{}

Result Tree::predict(Eigen::VectorXi datapoint)
{
	int pos = 0;
	Node* head = _treeNodes[pos];
	while (head->isLeaf())
	{
		pos = head->predict(datapoint, pos);
		head = _treeNodes.at(pos);
	}
	Result r;
	head->getResult(r);
	return r;
}

void Tree::train(Sample *sample)
{
	// for all the possible features, only numFeats features at each node
	// is calculated
	int numFeats = sample->getNumFeatures();
	Eigen::VectorXi samplesId = sample->getSelectedSamplesId();
	Sample *nodeSample = new Sample(sample, samplesId);
	_treeNodes[0] = new Node();
	_treeNodes[0]->_samples = nodeSample;
	// calculate the probability and gini
	_treeNodes[0]->computeNodeGini();
	for (int i = 0; i < _numNodes; ++i)
	{
		int parentId = (i - 1) / 2;
		// if current node's parent is null, continue
		if (_treeNodes[parentId] == nullptr)
			continue;
		// if current node's parent is a leaf, continue
		if (i > 0 and _treeNodes[parentId]->isLeaf())
			continue;
		// if maxDepth is reached, set current node as a leaf node
		if (i * 2 + 1 >= _numNodes)
		{
			_treeNodes[i]->createLeaf();
			continue;
		}
		// if current samples in this node is less than the threshold
		// set current node as a leaf node and continue
		if (_treeNodes[i]->_samples->getNumSelectedSamples() <= _minNumSamplesPerLeaf)
		{
			_treeNodes[i]->createLeaf();
			continue;
		}
		_treeNodes[i]->_samples->randomSampleFeatures();
		_treeNodes[i]->splitNodeByGini(_treeNodes, i, _giniThresh);
	}
}

void Tree::createNode(int nodeId, int featId, float thresh)
{
	_treeNodes[nodeId] = new Node();
	_treeNodes[nodeId]->setLeaf(false);
	_treeNodes[nodeId]->setFeatureId(featId);
	_treeNodes[nodeId]->getBestFeature();
}

void Tree::createLeaf(int nodeId, int classLabel, float prob)
{
	_treeNodes[nodeId] = new Node();
	_treeNodes[nodeId]->setLeaf(true);
	_treeNodes[nodeId]->setClass(classLabel);
	_treeNodes[nodeId]->setProb(prob);
}

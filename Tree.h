#pragma once

#include <iostream>
#include <cmath>
#include "Sample.h"
#include "Node.h"
#include "FeatureFactory.h"

class Tree
{
public:
	Tree(int maxDepth, int numFeatPerNode, int minNumSamplesPerLeaf, float giniThresh);
	void train(Sample *sample);
	Result predict(Eigen::VectorXi datapoint);
	inline std::vector<Node*> getTreeNodes() { return _treeNodes; }
	void createNode(int nodeId, int featId, float thresh);
	void createLeaf(int nodeId, int classLabel, float prob);
private:
	int _maxDepth;
	int _numNodes; // number of nodes = 2^_maxDepth - 1;
	int _minNumSamplesPerLeaf;
	int _numFeatPerNode;
	float _giniThresh;
	std::vector<Node*> _treeNodes;
};
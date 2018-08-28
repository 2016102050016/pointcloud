/*******************************************
This class holds the basic element of a tree
********************************************/

#pragma once

#include <vector>
#include "Sample.h"

struct Result
{
	float prob;
	int label;
};

class Node
{
public:
	Node();

	// the samples stored in this node
	Sample *_samples;
	// set the node as leaf
	inline void setLeaf(bool flag) { _isLeaf = flag; }
	inline bool isLeaf() { return _isLeaf; }

	// split the node into two child nodes or set it as a leaf node
	// based on the gini index of this node
	void splitNodeByGini(std::vector<Node*>& nodes, int nodeId, float threshGini);
	void computeNodeGini();
	float computeGini(Eigen::VectorXi& labelVec);

	// create a leaf node
	void createLeaf();

	// predict the data
	int predict(Eigen::VectorXi &cloud, int pointId);
	void getResult(Result &res);

	inline Features getBestFeature() { return _bestFeat; }
	inline void setFeatureId(int featId) { _featId = featId; }
	/*inline float getTresh() { return _thresh; }
	inline void setThresh(float thresh) { _thresh = thresh; }*/
	inline float getNodeGini() { return _gini; }
	inline int getClass() { return _class; }
	inline float getProb() { return _prob; }

	inline void setClass(int clas) { _class = clas; }
	inline void setProb(float prob) { _prob = prob; }

	// parameters for training
	float _gini;
	std::vector<float> _probs;

private:
	bool _isLeaf;
	Features _bestFeat;
	//float _thresh;
	int _class;
	float _prob;
	int _featId;
};
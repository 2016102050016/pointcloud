#pragma once

#include<math.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include"Tree.h"
#include"Sample.h"

class RandomForest
{
public:
	/*************************************************************
	numTrees: number of trees in this forest
	maxDepth: the max possible depth of each tree
	minSamplesPerLeaf: set the terminating condition for growing a tree
	giniThresh: the threshold at which a bag of samples should be split up
	**************************************************************/
	RandomForest(int numTrees, int maxDepth, int minSamplesPerLeaf, 
				 float giniThresh);
	// RandomForest(const char*modelPath);
	~RandomForest();
	
	/***************************************************************
	trainset:: the training set with each row representing a datapoint (dims: N*7)
	labels: the corresponding labels for each datapoint (dims: N*1)
	inidces: the indices of nearest neighbors for each datapoint (dims: N*k with k being
		     the number of nearest neighbors in a neighborhood)
	dists: the corresponding dists from the query point to its neighbors (dims: N*k)
	numClasses: number of classes, for this work it's 8
	numFeatsPerNode: number of features used at each node, the potential number of features
					 at each node is huge, so only a small fraction of that is used to limit
					 the computational effort
	**************************************************************/
	void train(Eigen::MatrixXf &trainset, Eigen::VectorXi &labels, Eigen::MatrixXi &indices,
			   Eigen::MatrixXf &dists, int numClasses, int numFeatsPerNode);
	/************************************************
	*sample: a single sample
	*response: the predict result
	*************************************************/
	void predict(float*sample, float&response);
	/************************************************
	*testset: the test set
	*SampleNum:the sample number in the testset
	*responses: the predict results
	*************************************************/
	void predict(Eigen::MatrixXf &dataset, int &res);
	/************************************************
	*path: the path to save the model
	*************************************************/
	//void saveModel(const char*path);
	/************************************************
	*path: the path to read the model
	*************************************************/
	//void readModel(const char*path);
private:
	int _numFeatsPerNode;  //the feature number used in a node while training
	int _numTrees;  //the number of trees
	int _maxDepth;  //the max depth which a tree can reach
	int _numClasses;  //the number of classes
	int _minSamplesPerLeaf;  //terminate condition the min samples in a node
	float _giniThresh;  //terminate condition the min information gain in a node
	std::vector<Tree*> _forest;//to store every tree
	Sample *_trainSample;  //hold the whole trainset and some other infomation
};

/***************************************************
This class is used for projecting a high dimensional
data to a real value (1d) so that simple comparison 
at a given node is possible
***************************************************/

#pragma once
#include "C:/Eigen/Eigen/Dense"
#include "Sample.h"

class FeatureFactory
{
public:

	/*FeatureFactory(const Sample &sample, int pointId);
	bool chooseProjector();
	std::vector<int> getSelectedPointId() { return _selectedPointId; }
	int getProjectionType() { return _projectionType; }*/
	FeatureFactory(Eigen::MatrixXf& neighborhood, int featType, int point1, int point2);
	bool computeFeature();

private:
	bool redColorDiff();
	bool greenColorDiff();
	bool blueColorDiff();
	bool xDiff();
	bool yDiff();
	bool zDiff();
	Eigen::MatrixXf _neighborhood;
	int _featType;
	int _point1;
	int _point2;
};




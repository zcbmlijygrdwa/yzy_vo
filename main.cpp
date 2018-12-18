/**
 * BA Example 
 * Author: Xiang Gao
 * Date: 2016.3
 * Email: gaoxiang12@mails.tsinghua.edu.cn
 * 
 * 在这个程序中，我们读取两张图像，进行特征匹配。然后根据匹配得到的特征，计算相机运动以及特征点的位置。这是一个典型的Bundle Adjustment，我们用g2o进行优化。
 */

// for std
#include <iostream>
// for opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <boost/concept_check.hpp>

using namespace cv;
using namespace std;

// 寻找两个图像中的对应点，像素坐标系
// 输入：img1, img2 两张图像
// 输出：points1, points2, 两组对应的2D点
int     findCorrespondingPoints( const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, cv::Mat& img1_with_features);
//const int MAX_FEATURES = 500;
const int MAX_FEATURES = 100;
// 相机内参
double cx = 325.5;
double cy = 253.5;
double fx = 518.0;
double fy = 519.0;

int main( int argc, char** argv )
{

    VideoCapture cap(1);
    if(!cap.isOpened())  // check if we succeeded
    {
        cout<<"camera not open"<<endl;
        return -1;
    }


    cv::Mat img1; 
    cv::Mat img2; 

    cv::Mat t_g = Mat(3,1, CV_64F, cvScalar(0.));;

    Mat img1_with_features;
    cap>>img1;

    while(cap.isOpened())
    {
        cap>>img2;

        // 找到对应点
        vector<cv::Point2f> pts1, pts2;
        if ( findCorrespondingPoints( img1, img2, pts1, pts2, img1_with_features) == false )
        {
            //imshow("img1", img1);
            //imshow("img2", img2);
            if(waitKey(30) >= 0) break;
            img2.copyTo(img1);
            cout<<"匹配点不够！"<<endl;
            continue;
        }
        //cout<<"找到了"<<pts1.size()<<"组对应特征点。"<<endl;
        //use epipolar constrain
        cv::Mat mask;
        cv::Mat e_mat;
        e_mat = cv::findEssentialMat(pts1,pts2,fx,cv::Point2d(cx,cy),cv::RANSAC, 0.999, 1.f,mask);
        //cout << "E:" << endl << e_mat/e_mat.at<double>(2,2) << endl;
        cv::Mat R, t;
        cv::recoverPose(e_mat, pts1, pts2, R, t,fx,cv::Point2d(cx,cy),mask);
        //end of using epipolar constrain
        //imshow("img1", img1);
        //imshow("img2", img2);
        imshow("img1_with_features", img1_with_features);

        t_g = t_g+t;
        cout<<"Pose[from epipolar constrain]: T = "<< std::fixed <<std::setprecision(2)<<t.at<double>(0,0)<<", "<<t.at<double>(0,1)<<", "<<t.at<double>(0,2)<<", T_G = "<<t_g.at<double>(0,0)<<", "<<t_g.at<double>(0,1)<<", "<<t_g.at<double>(0,2)<<endl; 

        if(waitKey(30) >= 0) break;

        img2.copyTo(img1);
    }
    return 0;
}


int     findCorrespondingPoints( const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, cv::Mat& img1_with_features)
{
    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
    vector<cv::KeyPoint> kp1, kp2;
    cv::Mat desp1, desp2;
    orb->detectAndCompute( img1, cv::Mat(), kp1, desp1 );
    orb->detectAndCompute( img2, cv::Mat(), kp2, desp2 );
    //cout<<"分别找到了"<<kp1.size()<<"和"<<kp2.size()<<"个特征点"<<endl;

    if(kp1.size()==0||kp2.size()==0)
    {
        return false;
    }


    drawKeypoints(img1,kp1, img1_with_features, Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create( "BruteForce-Hamming");

    double knn_match_ratio=0.8;
    vector< vector<cv::DMatch> > matches_knn;
    matcher->knnMatch( desp1, desp2, matches_knn, 2 );
    vector< cv::DMatch > matches;
    for ( size_t i=0; i<matches_knn.size(); i++ )
    {
        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
            matches.push_back( matches_knn[i][0] );
    }

    if (matches.size() <= 20) //匹配点太少
        return false;

    for ( auto m:matches )
    {
        points1.push_back( kp1[m.queryIdx].pt );
        points2.push_back( kp2[m.trainIdx].pt );
    }

    return true;
}

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

//for Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
// for opencv 
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <boost/concept_check.hpp>




using namespace cv;
using namespace std;
using namespace Eigen;

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

clock_t deltaTime = 0;
unsigned int frames = 0;
double  frameRate = 30;

Mat traj_image = Mat::zeros( 800, 800, CV_8UC1);

double clockToMilliseconds(clock_t ticks){
    // units/(units/time) => time (seconds) * 1000 = milliseconds
    return (ticks/(double)CLOCKS_PER_SEC)*1000.0;
}


void visulizePose2d(Mat& traj_image,Isometry3d& pose_in)
{
    Vector3d translation = pose_in.translation();

    double drawX = translation(0);
    double drawY = translation(2);
    drawX*=0.5;
    drawY*=0.5;
    drawX = (int)drawX+traj_image.cols/2;
    drawY = (int)drawY+traj_image.rows/2;
    //cout<<"drawX = "<<drawX<<", drawY = "<<drawY<<endl;
    Point drawP = Point(drawX,drawY);
    line(traj_image,drawP,drawP,Scalar(255,255,255),3,8);
    imshow("traj_image", traj_image);
}

int main( int argc, char** argv )
{

    VideoCapture cap;

    if(argc!=3)
    {
        cout<<"Useage:"<<endl<<"./yzy_vo cam [cameraIdex]"<<endl<<"./yzy_vo video [pathTovideo]"<<endl;
        return -1;
    }

    if(strcmp(argv[1],"video")==0)
    {
        cout<<"Run as video input mode"<<endl;
        cap = VideoCapture(argv[2]);
    }
    else if(strcmp(argv[1],"cam")==0)
    {
        int cameraIdx = atoi(argv[2]);

        cout<<"Run as web_cam mode, preparing camera["<<cameraIdx<<"]"<<endl;
        cap = VideoCapture(cameraIdx);
    }
    else
    {
        cout<<"Useage:"<<endl<<"./yzy_vo cam [cameraIdex]"<<endl<<"./yzy_vo video [pathTovideo]"<<endl;

        return -1;
    }

    //VideoCapture cap(1);
    if(!cap.isOpened())  // check if we succeeded
    {
        cout<<"camera not open"<<endl;
        return -1;
    }


    cv::Mat img1; 
    cv::Mat img2; 


    //create Isometry object to keep tracking of pose
    Isometry3d pose_g = Isometry3d::Identity();

    float resizeFactor = 0.3f;

    Mat img1_with_features;
    cap>>img1;

    resize(img1, img1, cv::Size(), resizeFactor, resizeFactor);
    while(cap.isOpened())
    {
        clock_t beginFrame = clock();

        cap>>img2;
        resize(img2, img2, cv::Size(), resizeFactor, resizeFactor);

        // 找到对应点
        vector<cv::Point2f> pts1, pts2;
        if ( findCorrespondingPoints( img1, img2, pts1, pts2, img1_with_features) == false )
        {
            //imshow("img1", img1);
            //imshow("img2", img2);
            if(waitKey(30) >= 0) break;
            img2.copyTo(img1);
            cout<<"Insufficient matching!"<<endl;
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





        //accumulating transformation
        Matrix3d rot_mat;
        Vector3d t_mat;
        cv::cv2eigen(R,rot_mat);
        //cout<<"rot_mat = "<<rot_mat<<endl;
        cv::cv2eigen(t,t_mat);
        //cout<<"t_mat.size() = "<<t_mat.size()<<endl;
        //cout<<"t_mat = "<<t_mat<<endl;
        pose_g.rotate(rot_mat);
        pose_g.pretranslate(t_mat);
        cout<<"pose_g = "<<endl<<pose_g.matrix()<<endl;
        //cout<<"pose_g.translation() = "<<pose_g.translation()<<endl;

        //draw 2d trajectory onto mat
        visulizePose2d(traj_image,pose_g);


        cout<<"FPS = "<<frameRate<<endl; 

        //if(waitKey(30) >= 0) break;
        waitKey(1);

        img2.copyTo(img1);
        clock_t endFrame = clock();
        deltaTime += endFrame - beginFrame;
        frames ++;
        //if you really want FPS
        if( clockToMilliseconds(deltaTime)>1000.0)
        { //every second
            frameRate = (double)frames*0.5 +  frameRate*0.5; //more stable
            frames = 0;
            deltaTime -= CLOCKS_PER_SEC;
        }
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

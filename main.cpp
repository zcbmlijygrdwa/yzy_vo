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
#include "opencv2/video/tracking.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <boost/concept_check.hpp>
// for g2o
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/types/sba/types_six_dof_expmap.h>



using namespace cv;
using namespace std;
using namespace Eigen;


void featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)   {

    //this function automatically gets rid of points for which tracking fails

    vector<float> err;
    Size winSize=Size(21,21);                                                               
    TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    int status_size = status.size();
    for( int i=0; i<status_size; i++)
    {  Point2f pt = points2.at(i- indexCorrection);
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))    {
            if((pt.x<0)||(pt.y<0))    {
                status.at(i) = 0;
            }
            points1.erase (points1.begin() + (i - indexCorrection));
            points2.erase (points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}


int     findCorrespondingPoints( const cv::Mat& img1, const cv::Mat& img2, vector<cv::Point2f>& points1, vector<cv::Point2f>& points2, cv::Mat& img1_with_features);
//const int MAX_FEATURES = 500;
const int MAX_FEATURES = 500;
// 相机内参
double cx = 239.961714;
double cy = 256.842130;
double fx = 814.660678;
double fy = 815.013833;

clock_t deltaTime = 0;
unsigned int frames = 0;
int frameCount = 0;
double  frameRate = 30;

Mat traj_image = Mat::zeros( 800, 800, CV_8UC1);

double clockToMilliseconds(clock_t ticks){
    // units/(units/time) => time (seconds) * 1000 = milliseconds
    return (ticks/(double)CLOCKS_PER_SEC)*1000.0;
}


void g2o_pose(vector<cv::Point2f>& pts1,vector<cv::Point2f>& pts2,Eigen::Isometry3d* pose);

void visulizePose2d(Mat& traj_image,Isometry3d& pose_in)
{
    Vector3d translation = pose_in.translation();

    double drawX = translation(0);
    double drawY = translation(2);
    //drawX*=50;
    //drawY*=50;
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

    if(!cap.isOpened())  // check if we succeeded
    {
        cout<<"camera not open"<<endl;
        return -1;
    }


    cv::Mat img1; 
    cv::Mat img2; 


    //create Isometry object to keep tracking of pose
    Isometry3d pose_global = Isometry3d::Identity();

    pose_global.rotate(AngleAxisd(3.141592653*0.5,Vector3d::UnitX()));

    vector<uchar> status;

    vector<Point2f> prevFeatures;
    vector<Point2f> currFeatures;



    Mat img1_with_features;
    cap>>img1;
    //resize(img1, img1, cv::Size(), resizeFactor, resizeFactor);

    cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_FEATURES);
    vector<cv::KeyPoint> kp1;
    cv::Mat desp1;
    orb->detectAndCompute( img1, cv::Mat(), kp1, desp1 );
    vector<Point2f> tempPoints1;
    cout<<"kp1.size() = "<<kp1.size()<<endl;
    for(auto tempKp:kp1)
    { 
        tempPoints1.push_back( tempKp.pt );
    }
    prevFeatures = tempPoints1;
    

    while(cap.isOpened())
    {
        clock_t beginFrame = clock();


        cap>>img2;
        if(frameCount%10!=0)
        {
            frameCount++;
            continue;
        }
        //resize(img2, img2, cv::Size(), resizeFactor, resizeFactor);

        //cx = img2.cols/2;
        //cy = img2.rows/2;
        //cout<<"cx = "<<cx<<endl;
        // 找到对应点
        vector<cv::Point2f> pts1, pts2;
        if ( findCorrespondingPoints( img1, img2, pts1, pts2, img1_with_features) == false )
        {
            //imshow("img1", img1);
            //imshow("img2", img2);
            img2.copyTo(img1);
            cout<<"Insufficient matching!"<<endl;
            continue;
        }
        //cout<<"找到了"<<pts1.size()<<"组对应特征点。"<<endl;

        vector<Point2f> currFeatures;
        currFeatures = pts2; 

        //feature tracking

        cout<<"prevFeatures.size() = "<<prevFeatures.size()<<endl;
        cout<<"currFeatures.size() = "<<currFeatures.size()<<endl;

        featureTracking(img1, img2, prevFeatures, currFeatures, status);

        cout<<"2prevFeatures.size() = "<<prevFeatures.size()<<endl;
        cout<<"2currFeatures.size() = "<<currFeatures.size()<<endl;

        //use epipolar constrain
        cv::Mat mask;
        cv::Mat e_mat;
        e_mat = cv::findEssentialMat(pts1,pts2,fx,cv::Point2f(cx,cy),cv::RANSAC, 0.999, 1.f,mask);
        //e_mat = cv::findEssentialMat(prevFeatures,currFeatures,fx,cv::Point2f(cx,cy),cv::RANSAC, 0.999, 1.f,mask);
        //cout << "E:" << endl << e_mat/e_mat.at<double>(2,2) << endl;
        cv::Mat R, t;
        cv::recoverPose(e_mat, pts1, pts2, R, t,fx,cv::Point2f(cx,cy),mask);
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
        //cout<<"t_mat = "<<t_mat<<endl;
        Isometry3d pose_temp = Isometry3d::Identity();
        pose_temp.rotate(rot_mat);
        pose_temp.pretranslate(t_mat);


        //*******************************
        //*********   G2O based  ********
        //*******************************
        //g2o_pose(pts1, pts2,&pose_temp);
        //*******************************


        //pose_global = pose_global*pose_temp;
        pose_global = pose_temp*pose_global;

        //pose_global.rotate(rot_mat);
        //pose_global.pretranslate(t_mat);



        //pose_global.rotate(pose_temp.rotation());
        //pose_global.pretranslate(pose_temp.translation());

        Vector3d ea = pose_temp.rotation().eulerAngles(0, 1, 2);
        cout<<"pose_temp[R t] = ["<<ea.transpose()<<","<<pose_temp.translation().transpose()<<"]"<<endl;

        ea = pose_global.rotation().eulerAngles(0, 1, 2);
        cout<<"pose_global[R t] = ["<<ea.transpose()<<","<<pose_global.translation().transpose()<<"]"<<endl;
        //draw 2globald trajectory onto mat
        visulizePose2d(traj_image,pose_global);


        cout<<"[frame"<<frameCount<<"]FPS = "<<frameRate<<endl; 

        //if(waitKey(30) >= 0) break;
        waitKey(1);



        img2.copyTo(img1);
        prevFeatures = currFeatures;

        clock_t endFrame = clock();
        deltaTime += endFrame - beginFrame;
        frames ++;
        frameCount++;
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


    drawKeypoints(img1,kp1, img1_with_features, Scalar::all(-1),DrawMatchesFlags::DEFAULT);

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

    cout<<"matches.size() = "<<matches.size()<<endl;
    if (matches.size() <= 20) //匹配点太少
        return false;

    for ( auto m:matches )
    {
        points1.push_back( kp1[m.queryIdx].pt );
        points2.push_back( kp2[m.trainIdx].pt );
    }

    return true;
}



void g2o_pose(vector<cv::Point2f>& pts1,vector<cv::Point2f>& pts2,Eigen::Isometry3d* pose)
{

    double percentage = 0.0;
    // 构造g2o中的图
    // 先构造求解器
    g2o::SparseOptimizer    optimizer;
    // 使用Cholmod中的线性方程求解器
    g2o::BlockSolver_6_3::LinearSolverType* linearSolver = new  g2o::LinearSolverCholmod<g2o::BlockSolver_6_3::PoseMatrixType> ();
    // 6*3 的参数
    // 6 X 3 matrix, why 6 X 3?
    g2o::BlockSolver_6_3* block_solver = new g2o::BlockSolver_6_3( std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType>(linearSolver) );
    // L-M 下降 
    // select a iteration strategy
    g2o::OptimizationAlgorithmLevenberg* algorithm = new g2o::OptimizationAlgorithmLevenberg( std::unique_ptr<g2o::BlockSolver_6_3>(block_solver) );

    optimizer.setAlgorithm( algorithm );
    optimizer.setVerbose( false );

    // 添加节点
    // 两个位姿节点
    for ( int i=0; i<2; i++ )
    {
        g2o::VertexSE3Expmap* v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if ( i == 0)
            v->setFixed( true ); // 第一个点固定为零
        // 预设值为单位Pose，因为我们不知道任何信息
        v->setEstimate( g2o::SE3Quat() );
        optimizer.addVertex( v );
    }
    // 很多个特征点的节点
    // 以第一帧为准

    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::VertexSBAPointXYZ* v = new g2o::VertexSBAPointXYZ();
        v->setId( 2 + i );
        // 由于深度不知道，只能把深度设置为1了
        double z = 1;
        double x = ( pts1[i].x - cx ) * z / fx;
        double y = ( pts1[i].y - cy ) * z / fy;
        v->setMarginalized(true);
        v->setEstimate( Eigen::Vector3d(x,y,z) );
        optimizer.addVertex( v );
    }

    // 准备相机参数
    g2o::CameraParameters* camera = new g2o::CameraParameters( fx, Eigen::Vector2d(cx, cy), 0 );
    camera->setId(0);
    optimizer.addParameter( camera );

    // 准备边
    // 第一帧
    vector<g2o::EdgeProjectXYZ2UV*> edges;
    for ( size_t i=0; i<pts1.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(0)) );
        edge->setMeasurement( Eigen::Vector2d(pts1[i].x, pts1[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0, 0);
        edge->setLevel(0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }
    // 第二帧
    for ( size_t i=0; i<pts2.size(); i++ )
    {
        g2o::EdgeProjectXYZ2UV*  edge = new g2o::EdgeProjectXYZ2UV();
        edge->setVertex( 0, dynamic_cast<g2o::VertexSBAPointXYZ*>   (optimizer.vertex(i+2)) );
        edge->setVertex( 1, dynamic_cast<g2o::VertexSE3Expmap*>     (optimizer.vertex(1)) );
        edge->setMeasurement( Eigen::Vector2d(pts2[i].x, pts2[i].y ) );
        edge->setInformation( Eigen::Matrix2d::Identity() );
        edge->setParameterId(0,0);
        edge->setLevel(0);
        // 核函数
        edge->setRobustKernel( new g2o::RobustKernelHuber() );
        optimizer.addEdge( edge );
        edges.push_back(edge);
    }

    while(percentage<0.999)
    {
        //cout<<"开始优化"<<endl;
        //optimizer.setVerbose(true);
        optimizer.initializeOptimization(0);
        optimizer.optimize(10);
        //cout<<"优化完毕"<<endl;

        //我们比较关心两帧之间的变换矩阵
        g2o::VertexSE3Expmap* v = dynamic_cast<g2o::VertexSE3Expmap*>( optimizer.vertex(1) );
        *pose = v->estimate();

        //Eigen::Isometry3d pose2 = v->estimate();
        //since there is no scale, unify the pose
        //cout<<"g2o Pose="<<endl<<pose->matrix()<<endl;

        // 估计inlier的个数
        int inliers = 0;
        int outliers = 0;
        for ( auto e:edges )
        {
            e->computeError();
            // chi2 就是 error*\Omega*error, 如果这个数很大，说明此边的值与其他边很不相符
            //cout<<"e->level() = "<<e->level()<<endl;
            if (e->chi2() > 1 )
            {
                //cout<<"error = "<<e->chi2()<<endl;
                //remove outliers   //https://github.com/RainerKuemmerle/g2o/issues/259
                if(e->level()==0)
                {
                    e->setLevel(1);
                    outliers++;
                }
            }
            else
            {
                inliers++;
            }
        }

        percentage = max(percentage,inliers/(double)(inliers+outliers));
        cout<<"["<<(percentage)<<"]inliers in total points: "<<inliers<<"/"<<inliers+outliers<<endl;


    }
}

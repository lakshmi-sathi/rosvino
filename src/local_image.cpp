#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <chrono>

//Params
std::string img_path;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "local_image");
  ros::NodeHandle nh;
  if(!nh.getParam("/local_image/inp_img", img_path)){
    img_path="/home/turtlebot/Downloads/pedestrian.jpg";
  }
  cv_bridge::CvImage cv_image;
  cv_image.image = cv::imread(img_path,cv::IMREAD_COLOR);//reads image from folder
  cv_image.encoding = "bgr8";
  sensor_msgs::Image ros_image;
  cv_image.toImageMsg(ros_image);//converts image from opencv matrix 2 ros message form

  ros::Publisher pub = nh.advertise<sensor_msgs::Image>("/local_image/image", 1);
  ros::Rate loop_rate(100);// creates a variable denoting 5hz interval

  while (nh.ok()) 
  {
    pub.publish(ros_image);
    auto start = std::chrono::high_resolution_clock::now();			        
    loop_rate.sleep();//sleeps for the time denoted by loop rate variable
    auto stop = std::chrono::high_resolution_clock::now();			        
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(stop - start);
    std::cout<<"\nDuration = "<< duration.count() <<std::endl;
  }
}

#include <ros/ros.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <chrono>

//Params
std::string vid_path;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "local_video");
  ros::NodeHandle nh;
  if(!nh.getParam("/local_video/inp_vid", vid_path)){
    vid_path="/home/turtlebot/Downloads/vid.mp4";
  }

  ros::Publisher vid_pub = nh.advertise<sensor_msgs::Image>("/local_video/video", 1);
  ros::Rate frame_rate(10);
  cv::VideoCapture vid(vid_path);
  if (!vid.isOpened()){
      std::cout<<"Failed to load video resource!\n";
  }
  cv::Mat frame;
  //vid.read(cv::OutputArray frame);

  while(nh.ok()){
      vid >> frame;
      if (frame.empty())
	      break;
      sensor_msgs::ImagePtr frame_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
      vid_pub.publish(frame_msg);  
      std::cout<<"Published\n";
      frame_rate.sleep();
  }

}
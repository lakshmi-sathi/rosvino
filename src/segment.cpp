#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <thread>
#include <sstream>
#include <inference_engine.hpp>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>


using namespace InferenceEngine;


//For Latency Timing and Throughput calculation
double start;
double stop;
float running_avg_latency = 0;
int fps_count = 0;
double start_time = 0; 
float fps = 0;
int latency_counter = 0;

//Node parameters
std::string device;
std::string network_loc;


//Class handling initializations and callback
class Segment{
	public:
	Segment(ros::NodeHandle &nh, Core &ie, CNNNetwork &network_reader, std::string &device){

    	//Publisher
		seg_result_labels = nh.advertise<sensor_msgs::Image>("/segment/result_labels",1); //for publishing image of class labels
		seg_result_probs = nh.advertise<sensor_msgs::Image>("/segment/result_probs",1); //for publishing image of class probabilites
    	
    	//Subscriber
		image_sub = nh.subscribe("/segment/input",1, &Segment::imageCallback, this);
		
		//Collect input layer info (needed later)
		InputsDataMap input_info(network_reader.getInputsInfo());
		InputInfo::Ptr& input_data = input_info.begin()->second;
		inputLayerName = input_info.begin()->first;
		
		//Set input layer settings
		input_data->setPrecision(Precision::U8);
		input_data->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
		input_data->getInputData()->setLayout(Layout::NCHW);

		//Collect output layer info
		OutputsDataMap output_info(network_reader.getOutputsInfo());
		DataPtr& output_data = output_info.begin()->second;
		outputLayerName = output_info.begin()->first;
		
		//Set output layer settings
		output_data->setPrecision(Precision::FP32);
		output_data->setLayout(Layout::NCHW);  //Change to BLOCKED for models with single channel outputs

		//Load network to device
		model_network = ie.LoadNetwork(network_reader, device);

		//Create Inference Request
		inferReq = model_network.CreateInferRequestPtr();
  	}

	//OpenCV image -> Blob pointer.
	static InferenceEngine::Blob::Ptr mat_to_blob(const cv::Mat &image) {
    InferenceEngine::TensorDesc tensor(InferenceEngine::Precision::U8,{1, (size_t)image.channels(), (size_t)image.size().height, (size_t)image.size().width},InferenceEngine::Layout::NHWC);
    return InferenceEngine::make_shared_blob<uint8_t>(tensor, image.data);
	}

	//OpenCV image -> Blob pointer -> Set blob to inference request.
	void frame_to_blob(const cv::Mat& image, InferRequest::Ptr& inferReq, const std::string& descriptor) {
    inferReq->SetBlob(descriptor, mat_to_blob(image));
	}

	//Run each time a new image is obtained
	void imageCallback(const sensor_msgs::Image::ConstPtr& image_msg){
		//Fetch image from image message of the subscribed topic
	    cv::Mat color_mat(image_msg->height,image_msg->width,CV_MAKETYPE(CV_8U,3),const_cast<uchar*>(&image_msg->data[0]), image_msg->step);
		
		//DEBUG LINES
	    //cv::cvtColor(color_mat,color_mat,cv::COLOR_BGR2RGB);
	    //cv::imwrite("/home/turtlebot/image.jpg", color_mat);
		/////////////
		
		//Image -> Blob pointer -> Set blob to inference request
		frame_to_blob(color_mat, inferReq, inputLayerName);

		start = ros::Time::now().toSec(); //start Latency timing
		
		inferReq->Infer(); //START INFERENCE//

		//Block execution proceed only once result ready.
		if (OK == inferReq->Wait(IInferRequest::WaitMode::RESULT_READY)) {
			//INFERENCE COMPLETE//
			stop = ros::Time::now().toSec(); //Take Latency timing
			double duration = stop - start; 
			fps_count++;
			latency_counter++;

			//Calculate running average latency
			if(latency_counter%4==0){
			    running_avg_latency += duration;
			    running_avg_latency = (float) running_avg_latency / 4;
			    std::cout<<"- Running Average Latency = "<< running_avg_latency << " secs -\n";
			    running_avg_latency = 0;
			} else {
			    running_avg_latency += duration;
			}

			//Fetch the result blob associated with the inference request
			try{
				output_blob = inferReq->GetBlob(outputLayerName);
				output_data = output_blob->buffer().as<float*>();
			}
			catch(const std::exception& e){
				ROS_ERROR("%s",e.what());
				std::cout<<"Error retrieving inference results. More:" << e.what() << std::endl;
				//return -1;
			}

			//Fetch size information of the result in the blob
			size_t N = output_blob->getTensorDesc().getDims().at(0); 
       	 	size_t C, H, W;
	        size_t output_blob_shape_size = output_blob->getTensorDesc().getDims().size();
			if (output_blob_shape_size == 3) {
				C = 1;
				H = output_blob->getTensorDesc().getDims().at(1);
				W = output_blob->getTensorDesc().getDims().at(2);
			} else if (output_blob_shape_size == 4) {
				C = output_blob->getTensorDesc().getDims().at(1);
	          	H = output_blob->getTensorDesc().getDims().at(2);
	    		W = output_blob->getTensorDesc().getDims().at(3);
			} else {
				std::cout << "\n Matrix size 3 or 4 only Supported \n";
			}

			//Size to increment the blob pointer for traversing the result image-by-image
			size_t image_stride = W*H*C;

			// Iterating over all images
			for (size_t image = 0; image < N; ++image) {
				//Initialize vectors to hold the result
				std::vector<std::vector<int>> outArrayClasses(H, std::vector<int>(W, 0));        //Class label of each pixel
				std::vector<std::vector<float>> outArrayProb(H, std::vector<float>(W, 0.));      //Class probabilities of each pixel
				// Iterating over each pixel (across rows and columns)	
				for (size_t w = 0; w < W; ++w) {
					for (size_t h = 0; h < H; ++h) {
						// number of channels = 1 means that the output is already ArgMax'ed
						if (C == 1) {
							outArrayClasses[h][w] = static_cast<int>(output_data[image_stride * image + W * h + w]);
						} else {
							//Traverse over each channel and fill outArrayClasses with the predicted class
							//and outArrayProb with the corresponding probability
							for (int ch = 0; ch < C; ++ch) {
								auto data = output_data[image_stride * image + W * H * ch + W * h + w];   //Accessing the data using the blob pointer
								//Store just the largest class probability and the corresponding class
								if (data > outArrayProb[h][w]) {
									outArrayClasses[h][w] = ch;  
									outArrayProb[h][w] = data; 
								}
							}
						}
					}
				}
				
				//Scale each class label for easy visualisation
				//Visualizable version of Class labels and Class probabilities
				std::vector<std::vector<int>> outArrayClasses_image(H, std::vector<int>(W, 0));
				std::vector<std::vector<float>> outArrayProb_image(H, std::vector<float>(W, 0.));
				for(unsigned int i = 0; i < outArrayClasses_image.size(); i++)
					for(unsigned int j = 0; j < outArrayClasses_image[0].size(); j++){
						outArrayClasses_image[i][j]=(unsigned int) 85*outArrayClasses[i][j]; //Scaling pixels from 0-3 to 0-255
						outArrayProb_image[i][j]=(float) 255*outArrayProb[i][j];      //Scaling pixels from 0<p<1 to 0-255
					}

				//Conversion of class labels output image from vector to opencv matrix
				cv::Mat output_label_image(0, outArrayClasses_image[0].size(), CV_8UC1); //cv::DataType<int>::type);
				cv::Mat output_prob_image(0, outArrayProb_image[0].size(), CV_32FC1); //cv::DataType<float>::type);
				for (unsigned int i = 0; i < outArrayClasses_image.size(); ++i)
				{
  					// Make a temporary cv::Mat row for each label and prob images and push into final cv::Mat image. cv::DataType<int>::type
  					cv::Mat label_row(1, outArrayClasses_image[0].size(), cv::DataType<int>::type, outArrayClasses_image[i].data());
					cv::Mat prob_row(1, outArrayProb_image[0].size(), CV_32FC1, outArrayProb_image[i].data());
   					output_label_image.push_back(label_row);
					output_prob_image.push_back(prob_row);
				}
				
				//DEBUG LINES
				std::string prob = "/home/turtlebot/prob.png";
				std::string label = "/home/turtlebot/label.png";
				cv::imwrite(prob, output_prob_image);
				cv::imwrite(label, output_label_image);
				output_label_image = cv::imread(label);
				output_prob_image = cv::imread(prob);
				//cv::cvtColor(output_label_image, output_label_image, CV_BGR2GRAY);
				//cv::cvtColor(output_prob_image, output_prob_image, CV_BGR2GRAY);
				/////////////

				////Publishing the result
				sensor_msgs::ImagePtr output_class_label_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", output_label_image).toImageMsg();
				sensor_msgs::ImagePtr output_class_prob_msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", output_prob_image).toImageMsg();
				seg_result_labels.publish(output_class_label_msg);
				seg_result_probs.publish(output_class_prob_msg);

				//Calculate inferences per second
				double stop_time = ros::Time::now().toSec();
				fps = (stop_time - start_time > 1)?fps_count:fps;
				if(stop_time - start_time > 1){
					std::cout<<"- Inferences per second = "<< fps << " -\n";
					start_time = ros::Time::now().toSec();
					fps_count = 0;
				}
			}
		}
		//Callback Code Ends Here//
	}
	private:
  	ros::NodeHandle nh; 
  	ros::Publisher seg_result_labels, seg_result_probs;
  	ros::Subscriber image_sub;
	InferRequest::Ptr inferReq;            //For holding the inference request
	ExecutableNetwork model_network;       //For loading network to device
	std::string inputLayerName;        
	std::string outputLayerName;
	float *output_data;                    //For fetching results into
	Blob::Ptr output_blob;
	sensor_msgs::Image output_image_msg;   //For output image message
	cv_bridge::CvImage out_image;
};


//*************************************************************
//
//                         MAIN
//
//*************************************************************

int main(int argc, char **argv) {
	//Initialise node
	ros::init(argc, argv, "segment");
	ros::NodeHandle nh;
	Core ie;

	//Fetch input parameters for the node at launch
	if(!nh.getParam("/segment/target_device", device)){
		device="MYRIAD";
	}
	if(!nh.getParam("/segment/network", network_loc)){
		network_loc = "/home/turtlebot/catkin_ws/src/rosvino/models/segmentation/road-segmentation-adas-0001/road-segmentation-adas-0001.xml";
	}

	//Read the network from IR files (.xml - network file & .bin - weights file)
	CNNNetwork network_reader = ie.ReadNetwork(network_loc);
    
	//Initializing the object for the segment class
	Segment segment_this(nh, ie, network_reader, device);

	//The running loop for the node that calls the image callback function each time an image is received.
	///////////////////
	ros::spin();
	///////////////////
}
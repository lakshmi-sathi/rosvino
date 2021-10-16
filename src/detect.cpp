#include <ros/ros.h>
#include <iostream>
#include <thread>
#include <fstream>
#include <inference_engine.hpp>
#include <rosvino/Object.h>
#include <rosvino/Objects.h>
#include <sensor_msgs/Image.h>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>

using namespace InferenceEngine;


//Timing and CPU usgae
double start;
double stop;
float running_avg_latency = 0;
int count = 0;
int latency_counter = 0;
double start_time = 0; 
float fps = 0;

//Node parameters
std::string device;
float confidence_thresh;
std::string network_loc;


//Class containing subscriber publisher and callback
class Detect{
	public:
	Detect(ros::NodeHandle &nh, Core &ie, CNNNetwork &network_reader, std::string &device){
    	//Publisher
    	det_results = nh.advertise<rosvino::Objects>("/detect/det_results",1);

    	//Subscriber
		image_sub = nh.subscribe("/detect/input", 1, &Detect::imageCallback, this);

		//taking input layer info
		InputsDataMap input_info(network_reader.getInputsInfo());
		InputInfo::Ptr& input_data = input_info.begin()->second;
		inputLayerName = input_info.begin()->first;
		
		//setting input layer settings
		input_data->setPrecision(Precision::U8);
		input_data->getPreProcess().setResizeAlgorithm(ResizeAlgorithm::RESIZE_BILINEAR);
		input_data->getInputData()->setLayout(Layout::NHWC);

		//taking output layer info
		OutputsDataMap output_info(network_reader.getOutputsInfo());
		DataPtr& output_data = output_info.begin()->second;
		outputLayerName = output_info.begin()->first;
		//num_classes = network_reader.getLayerByName(outputLayerName.c_str())->GetParamAsInt("num_classes");
		output_dimension = output_data->getTensorDesc().getDims();
		results_number = output_dimension[2];
		object_size = output_dimension[3];
		if ((object_size != 7 || output_dimension.size() != 4)) {
			ROS_ERROR("There is a problem with output dimension");
		}

		//Setting output layer settings
		output_data->setPrecision(Precision::FP32);
		output_data->setLayout(Layout::NCHW);

		//Load Network to device
	    model_network = ie.LoadNetwork(network_reader, device);
		
		//Create Inference Request
		inferReq = model_network.CreateInferRequestPtr();
  	}

	//Image matrix to blob conversion
	// /OpenCV mat to blob
	static InferenceEngine::Blob::Ptr mat_to_blob(const cv::Mat &image) {
	    InferenceEngine::TensorDesc tensor(InferenceEngine::Precision::U8,{1, (size_t)image.channels(), (size_t)image.size().height, (size_t)image.size().width},InferenceEngine::Layout::NHWC);
	    return InferenceEngine::make_shared_blob<uint8_t>(tensor, image.data);
	}

	//Image to blob and associate with inference request
	void frame_to_blob(const cv::Mat& image, InferRequest::Ptr& inferReq, const std::string& descriptor) {
	    inferReq->SetBlob(descriptor, mat_to_blob(image));
	}

	//Callback
	void imageCallback(const sensor_msgs::Image::ConstPtr& image_msg){
	    cv::Mat color_mat(image_msg->height,image_msg->width,CV_MAKETYPE(CV_8U,3),const_cast<uchar*>(&image_msg->data[0]), image_msg->step);
	    //cv::cvtColor(color_mat,color_mat,cv::COLOR_BGR2RGB);
		width  = (size_t)color_mat.size().width;
        height = (size_t)color_mat.size().height;
		frame_to_blob(color_mat, inferReq, inputLayerName);
	    start = ros::Time::now().toSec(); //Latency timing

		inferReq->Infer(); //Inference starts

		//Wait till result ready and then take
	    if (OK == inferReq->Wait(IInferRequest::WaitMode::RESULT_READY)) {
			//Take Latency calculation
			stop = ros::Time::now().toSec();
			double duration = stop - start; 
			count++;
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
			
			try{
			    //Fetch the results associated with the inference request
				compute_results = inferReq->GetBlob(outputLayerName)->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
			}
			catch(const std::exception& e){
				ROS_ERROR("%s",e.what());
				std::cout<<"Error retrieving inference results. More:" << e.what() << std::endl;
			}

	    	for (int i = 0; i < results_number; i++) {
				//Extract the individual result values from the aggregated result
	    	    float result_id = compute_results[i * object_size + 0];
			    int result_label = static_cast<int>(compute_results[i * object_size + 1]);
	    	    float result_confidence = compute_results[i * object_size + 2];
	    	    float result_xmin = compute_results[i * object_size + 3];
	    	    float result_ymin = compute_results[i * object_size + 4];
	    	    float result_xmax = compute_results[i * object_size + 5];
	    	    float result_ymax = compute_results[i * object_size + 6];
				//Print out the results
				if(result_label && result_confidence > confidence_thresh){
					printf("------DETECTION------\nLabel = %d\nConfidence = %.1f\nXmin = %.1f\nYmin = %.1f\nWidth = %.1f\nHeight = %.1f\n---------------------", result_label, result_confidence*100, result_xmin*width, result_ymin*height, (result_xmax-result_xmin)*width, (result_ymax-result_ymin)*height);
				}

				if (result_confidence > confidence_thresh){
					//Load the results to message object
					result_obj.label= result_label;
					result_obj.confidence=result_confidence;
					result_obj.x=result_xmin;
					result_obj.y=result_ymin;
					result_obj.width=result_xmax-result_xmin;
					result_obj.height=result_ymax-result_ymin;
					results.objects.push_back(result_obj);

					//Publish the results obtained/
					try{
						std::cout<<"\nPublishing result\n";
						results.header.stamp=ros::Time::now();
						det_results.publish(results);
						results.objects.clear();
					}
					catch(const std::exception& e){
						ROS_ERROR("%s",e.what());
						std::cout<<"Error publishing inference result. More:" << e.what() << std::endl;
					}

					//Calculate inferences per second
					double stop_time = ros::Time::now().toSec();
					fps = (stop_time - start_time > 1)?count:fps;
					if(stop_time - start_time > 1){
						std::cout<<"\n- Inferences per second = "<< fps << " -\n";
						start_time = ros::Time::now().toSec();
						count = 0;
					}
				
				}
			}


		}
		//Callback Code Ends Here//
	}

	private:
  	ros::NodeHandle nh; 
  	ros::Publisher det_results;
  	ros::Subscriber image_sub;
	InferRequest::Ptr inferReq;            //For holding the inference request
	ExecutableNetwork model_network;       //For loading network to device
	std::string inputLayerName;        
	std::string outputLayerName;
	size_t width, height;
	float *compute_results;                //For fetching results into
	rosvino::Object result_obj;            //output msg classes
	rosvino::Objects results;
	SizeVector output_dimension;           //To hold information about output
	int results_number;
	int object_size;
};


//*************************************************************
//
//                         MAIN
//
//*************************************************************

int main(int argc, char **argv) {
	//Initialise
	ros::init(argc, argv, "detect");
	ros::NodeHandle nh;
	Core ie;

	//Fetch params
	if(!nh.getParam("/detect/threshold", confidence_thresh)){
		confidence_thresh=0.5;
	}
	if(!nh.getParam("/detect/target_device", device)){
		device="MYRIAD";
	}
	if(!nh.getParam("/detect/network", network_loc)){
		network_loc = "/opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/intel/pedestrian-detection-adas-0002/FP16/pedestrian-detection-adas-0002.xml";
	}
	
	//Read the network from IR files
	CNNNetwork network_reader = ie.ReadNetwork(network_loc);
	
	//Creating the object for the class
	Detect detect(nh, ie, network_reader, device);

	//The running loop for the node
	//*************************************************************
	ros::spin();
	//*************************************************************
}


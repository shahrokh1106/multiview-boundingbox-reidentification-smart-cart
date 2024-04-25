
// Created by Shahrokh Heidari
// March 1st, 2024
// Testing and Getting features from dinos' detection model with MobNet backbone

#include <optional>
#include <filesystem>
#include <iostream>
#include <random>
#include "spdlog/spdlog.h"
#include "catch2/catch_test_macros.hpp"
#include "fmt/format.h"

#include "edgetpu.h"

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/kernels/register_ref.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include "opencv2/opencv.hpp"
#include <Eigen/Dense>
#include "nlohmann/json.hpp"
#include <fstream>
#include "profiling/rolling_average.hpp"
#include "profiling/stopwatch.hpp"
#include <cmath>


cv::Mat ReadRGB(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    cv::Mat img(320, 320, CV_8UC3);
    file.read(reinterpret_cast<char*>(img.data), 320 * 320 * 3 * sizeof(uint8_t));
    file.close();
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    return img;
}

cv::Mat DrawBbox(cv::Mat IMG, std::vector<float> Bbox, cv::Scalar color) {
    cv::Point pt1(Bbox[0], Bbox[1]);    // Top-left corner
    cv::Point pt2(Bbox[2], Bbox[3]);  // Bottom-right corner
    cv::Point center((Bbox[0]+ Bbox[2])/2, (Bbox[1] + Bbox[3])/2); // Get the center of the bbox
    // Draw the rectangle on the image
    cv::rectangle(IMG, pt1, pt2, color, 2);
    cv::circle(IMG, center, 3, color, -1, 2);
    return IMG;
}
cv::Mat DrawLine(cv::Mat IMG, Eigen::VectorXf Coeff, cv::Scalar color) {
    float x1 = 0.0;
    float y1 = Coeff(2) / -Coeff(1);
    float x2 = 1320;
    float y2 = (Coeff(0) * x2 + Coeff(2)) / -Coeff(1);
    cv::Point pt1(x1, y1);
    cv::Point pt2(x2, y2);
    cv::line(IMG, pt1, pt2, color, 2, 2);
    return IMG;
}

// Loading calibration parameters for ech camera view
const std::string json_filename_cam0 = "/opt/imagr/etc/test_data/shah_test/calib_data_cam0.json";
const std::string json_filename_cam1 = "/opt/imagr/etc/test_data/shah_test/calib_data_cam1.json";
const std::string json_filename_cam2 = "/opt/imagr/etc/test_data/shah_test/calib_data_cam2.json";
std::ifstream f0(json_filename_cam0);
std::ifstream f1(json_filename_cam1);
std::ifstream f2(json_filename_cam2);
auto CalibDataCam0 = nlohmann::json::parse(f0);
auto CalibDataCam1 = nlohmann::json::parse(f1);
auto CalibDataCam2 = nlohmann::json::parse(f2);
struct CalibStruct {
    std::vector<std::vector<float>> K0 = CalibDataCam0["K"].get<std::vector<std::vector<float>>>();
    std::vector<std::vector<float>> RT0 = CalibDataCam0["RT"].get<std::vector<std::vector<float>>>();
    std::vector<std::vector<float>> InvRT0 = CalibDataCam0["invRT"].get<std::vector<std::vector<float>>>();
    std::vector<std::vector<float>> K1 = CalibDataCam1["K"].get<std::vector<std::vector<float>>>();
    std::vector<std::vector<float>> RT1 = CalibDataCam1["RT"].get<std::vector<std::vector<float>>>();
    std::vector<std::vector<float>> InvRT1 = CalibDataCam1["invRT"].get<std::vector<std::vector<float>>>();
    std::vector<std::vector<float>> K2 = CalibDataCam2["K"].get<std::vector<std::vector<float>>>();
    std::vector<std::vector<float>> RT2 = CalibDataCam2["RT"].get<std::vector<std::vector<float>>>();
    std::vector<std::vector<float>> InvRT2 = CalibDataCam2["invRT"].get<std::vector<std::vector<float>>>();
} Calib;



Eigen::VectorXf ComputeEpipolarLine(CalibStruct Calib, std::vector<float> Bbox, std::string CamName1, std::string CamName2) {
    // To load the calib parameters of the first camera view
    float Ka[3][4];
    float RTa[4][4];
    float InvRTa[4][4];
    // To load the calib parameters of the second camera view
    float Kb[3][4];
    float RTb[4][4];
    float InvRTb[4][4];

    if (CamName1 == "cam0") {
        for (int i = 0; i < 3;i++) {
            for (int j = 0; j < 4;j++) {
                Ka[i][j] = Calib.K0[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                RTa[i][j] = Calib.RT0[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                InvRTa[i][j] = Calib.InvRT0[i][j];
            }
        }
    }
    if (CamName1 == "cam1") {
        for (int i = 0; i < 3;i++) {
            for (int j = 0; j < 4;j++) {
                Ka[i][j] = Calib.K1[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                RTa[i][j] = Calib.RT1[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                InvRTa[i][j] = Calib.InvRT1[i][j];
            }
        }
    }
    if (CamName1 == "cam2") {
        for (int i = 0; i < 3;i++) {
            for (int j = 0; j < 4;j++) {
                Ka[i][j] = Calib.K2[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                RTa[i][j] = Calib.RT2[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                InvRTa[i][j] = Calib.InvRT2[i][j];
            }
        }
    }

    if (CamName2 == "cam0") {
        for (int i = 0; i < 3;i++) {
            for (int j = 0; j < 4;j++) {
                Kb[i][j] = Calib.K0[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                RTb[i][j] = Calib.RT0[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                InvRTb[i][j] = Calib.InvRT0[i][j];
            }
        }
    }
    if (CamName2 == "cam1") {
        for (int i = 0; i < 3;i++) {
            for (int j = 0; j < 4;j++) {
                Kb[i][j] = Calib.K1[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                RTb[i][j] = Calib.RT1[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                InvRTb[i][j] = Calib.InvRT1[i][j];
            }
        }
    }
    if (CamName2 == "cam2") {
        for (int i = 0; i < 3;i++) {
            for (int j = 0; j < 4;j++) {
                Kb[i][j] = Calib.K2[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                RTb[i][j] = Calib.RT2[i][j];
            }
        }
        for (int i = 0; i < 4;i++) {
            for (int j = 0; j < 4;j++) {
                InvRTb[i][j] = Calib.InvRT2[i][j];
            }
        }
    }
    // Now, Ka,RTa, and InvRTa are the calib parameters of the first camera, and Kb, RTb, and InvRTb are the calib parameters of the second camera, first and second cameras could be any camera views (0, 1 , or 2)

    // Getting the transformation matrix: from the first camera to the second one
    float Cam_a_to_cam_b[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            Cam_a_to_cam_b[i][j] = 0;
            for (int k = 0; k < 4; ++k) {
                Cam_a_to_cam_b[i][j] += RTb[i][k] * InvRTa[k][j];
            }
        }
    }
    // Getting the rotation matrix: from the first camera to the second one
    float R12[3][3];
    for (int i = 0; i < 3;i++) {
        for (int j = 0; j < 3;j++) {
            R12[i][j] = Cam_a_to_cam_b[i][j];
        }
    }
    // Getting the translition vector: from the first camera to the second one
    float T12[3][1] = { {Cam_a_to_cam_b[0][3]},{Cam_a_to_cam_b[1][3]},{Cam_a_to_cam_b[2][3]} };

    // Changing the translition vector to a product matrix 
    float T12_Product_Matrix[3][3] = { {0.0,0.0,0.0},{0.0,0.0,0.0},{0.0,0.0,0.0} };
    T12_Product_Matrix[0][1] = -T12[2][0];
    T12_Product_Matrix[0][2] = T12[1][0];
    T12_Product_Matrix[1][0] = T12[2][0];
    T12_Product_Matrix[1][2] = -T12[0][0];
    T12_Product_Matrix[2][0] = -T12[1][0];
    T12_Product_Matrix[2][1] = T12[0][0];

    // Multiplying the translation product matrix by the rotation matrix to have the essential matrix (it's a transformation from the first camera to the second one but in world coordinate)
    float EssentialMatrix[3][3];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            EssentialMatrix[i][j] = 0;
            for (int k = 0; k < 3; ++k) {
                EssentialMatrix[i][j] += T12_Product_Matrix[i][k] * R12[k][j];
            }
        }
    }

    // This is just to get 3by3 camera matrices
    Eigen::MatrixXf KKa(3, 3);
    Eigen::MatrixXf KKb(3, 3);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            KKa(i, j) = Ka[i][j];
            KKb(i, j) = Kb[i][j];
        }
    }

    // This is just to get 3by3 essential matrices
    Eigen::MatrixXf EssentialMatrixx(3, 3);
    for (int i = 0; i < 3;i++) {
        for (int j = 0; j < 3;j++) {
            EssentialMatrixx(i, j) = EssentialMatrix[i][j];
        }
    }

    // Here we compute a transformation from the first camera to the second one that works based on pixels
    Eigen::MatrixXf FundamentalMatrix(3, 3);
    FundamentalMatrix = KKb.inverse().transpose() * EssentialMatrixx * KKa.inverse();

    // Once we have the fundamental matrix, for any point in the first camera view, we can compute the corresponding eppipolar line in the seocnd. Here our pixel is actually the center of the bounding box
    Eigen::MatrixXf center(3, 1);
    center(0, 0) = (Bbox[0] + Bbox[2]) / 2;
    center(1, 0) = (Bbox[1] + Bbox[3]) / 2;
    center(2, 0) = 1.0;
    Eigen::VectorXf Coeff(3);
    Coeff = FundamentalMatrix * center;

    // Coeff has three values showing the epipolar line parameters
    return Coeff;
}

float GetLineDistance(Eigen::VectorXf Coeff, std::vector<float> Bbox) {
    Eigen::VectorXf Center(2);
    Center(0) = (Bbox[0] + Bbox[2]) / 2;
    Center(1) = (Bbox[1] + Bbox[3]) / 2;
    return abs((Coeff(0) * Center(0)) + (Coeff(1) * Center(1)) + Coeff(2)) / sqrt((Coeff(0) * Coeff(0)) + (Coeff(1) * Coeff(1)));
}

cv::Mat ShowEppipolars(cv::Mat IMG0, cv::Mat IMG1, cv::Mat IMG2, CalibStruct Calib, std::vector<float> Bbox0, std::vector<float> Bbox1, std::vector<float> Bbox2, float EpiCost, float EmbCost, float TotalCost) {
    cv::Scalar color0(255, 0, 0);
    cv::Scalar color1(0, 255, 0);
    cv::Scalar color2(0, 0, 255);

    cv::Mat IMG0bbox = DrawBbox(IMG0.clone(), Bbox0, color0);
    cv::Mat IMG1bbox = DrawBbox(IMG1.clone(), Bbox1, color1);
    cv::Mat IMG2bbox = DrawBbox(IMG2.clone(), Bbox2, color2);

    Eigen::VectorXf Coeff01(3);
    Eigen::VectorXf Coeff02(3);
    Coeff01 = ComputeEpipolarLine(Calib, Bbox0, "cam0", "cam1");
    Coeff02 = ComputeEpipolarLine(Calib, Bbox0, "cam0", "cam2");
    cv::Mat IMGline01 = DrawLine(IMG1bbox, Coeff01, color0);
    cv::Mat IMGline02 = DrawLine(IMG2bbox, Coeff02, color0);

    Eigen::VectorXf Coeff10(3);
    Eigen::VectorXf Coeff12(3);
    Coeff10 = ComputeEpipolarLine(Calib, Bbox1, "cam1", "cam0");
    Coeff12 = ComputeEpipolarLine(Calib, Bbox1, "cam1", "cam2");
    cv::Mat IMGline10 = DrawLine(IMG0bbox, Coeff10, color1);
    cv::Mat IMGline12 = DrawLine(IMGline02, Coeff12, color1);

    Eigen::VectorXf Coeff20(3);
    Eigen::VectorXf Coeff21(3);
    Coeff20 = ComputeEpipolarLine(Calib, Bbox2, "cam2", "cam0");
    Coeff21 = ComputeEpipolarLine(Calib, Bbox2, "cam2", "cam1");
    cv::Mat IMGline20 = DrawLine(IMGline10, Coeff20, color2);
    cv::Mat IMGline21 = DrawLine(IMGline01, Coeff21, color2);


    cv::putText(IMGline21, "Total Epi Cost:  "+std::to_string(EpiCost), cv::Point(10, 270), cv::FONT_HERSHEY_DUPLEX, 0.4, color1, 1);
    cv::putText(IMGline21, "Total Emb Cost:  "+std::to_string(EmbCost), cv::Point(10, 290), cv::FONT_HERSHEY_DUPLEX, 0.4, color1, 1);
    cv::putText(IMGline21, "Total Emb Cost:  "+std::to_string(TotalCost), cv::Point(10, 310), cv::FONT_HERSHEY_DUPLEX, 0.4, color1, 1);

    cv::Mat concatenated1;
    cv::hconcat(IMGline20, IMGline21, concatenated1);
    cv::Mat concatenated2;
    cv::hconcat(concatenated1, IMGline12, concatenated2);
    //cv::imshow("Epipolar Lines", concatenated2);
    //int k = cv::waitKey(0);
    return concatenated2;
}
cv::Mat ShowBBoxAll(cv::Mat IMG0, cv::Mat IMG1, cv::Mat IMG2, std::vector<std::vector<float>> Bbox0, std::vector<std::vector<float>> Bbox1, std::vector<std::vector<float>> Bbox2) {
    int num_Bbox0 = Bbox0.size();
    int num_Bbox1 = Bbox1.size();
    int num_Bbox2 = Bbox2.size();
    cv::Scalar color(0, 0, 255);
    for (int i = 0; i < num_Bbox0;i++) {
        cv::Mat IMG0bbox = DrawBbox(IMG0, Bbox0[i], color);
    }
    for (int i = 0; i < num_Bbox1;i++) {
        cv::Mat IMG1bbox = DrawBbox(IMG1, Bbox1[i], color);
    }
    for (int i = 0; i < num_Bbox2;i++) {
        cv::Mat IMG2bbox = DrawBbox(IMG2, Bbox2[i], color);
    }
    cv::Mat concatenated1;
    cv::hconcat(IMG0, IMG1, concatenated1);
    cv::Mat concatenated2;
    cv::hconcat(concatenated1, IMG2, concatenated2);
    //cv::imshow("Epipolar Lines", concatenated2);
    //int k = cv::waitKey(0);
    return concatenated2;
}


struct MobNetBackbone_t {
  std::string model_filename;
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
  std::unique_ptr<tflite::Interpreter> interpreter;

  int model_width;
  int model_height;
  int model_channels;

  bool model_loaded{false};
  bool edgetpu_connected{false};
  bool interpreter_built{false};
  bool interpreter_configured{false};

  bool debug_logging{false};

  imagr::profiling::rolling_average_t copy_times{200};
  imagr::profiling::rolling_average_t infer_times{200};
  imagr::profiling::rolling_average_t process_times{200};

  auto load_model() -> bool {
    if (model_filename.empty()) {
      spdlog::error("Object detector constructed with no model filename.");
      return false;
    }

    model = tflite::FlatBufferModel::BuildFromFile(model_filename.c_str()); 

    // Check error: load failure
    if (model == nullptr) {
      spdlog::error("Object detector failed to load tflite model {}.", model_filename);
      return false;
    }

    return true;
  }

  auto connect_edgetpu() -> bool {
    edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();

    // Check error: hardware failure
    if (edgetpu_context == nullptr) {
      spdlog::error("Object detector failed to connect edgetpu.");
      return false;
    }

    return true;
  }

  auto connect_edgetpu(edgetpu::DeviceType device_type, std::string device_name) -> bool {
    edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice(device_type, device_name);

    // Check error: hardware failure
    if (edgetpu_context == nullptr) {
      spdlog::error("Object detector failed to connect edgetpu.");
      return false;
    }

    return true;
  }

  auto build_interpreter() -> bool {
    if (model == nullptr) return false;

    tflite::ops::builtin::BuiltinOpResolver resolver;
    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

    tflite::InterpreterBuilder builder(*model, resolver);
    builder.PreserveAllTensorsExperimental();
    auto builder_status = builder(&interpreter);


    // Check error: build failure
    if (builder_status != kTfLiteOk) {
      spdlog::error("Object detector failed to build tensorflow lite interpreter.");
      return false;
    }

    return true;
  }

  auto configure_interpreter() -> bool {
    if (interpreter == nullptr) return false;

    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context.get());
    interpreter->SetNumThreads(1);

    auto allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      spdlog::error("Object detector failed to allocate tensors.");
      return false;
    }

    return true;
  }

  MobNetBackbone_t(std::string model_filename) : model_filename(model_filename) {
    model_loaded = load_model();
    edgetpu_connected = connect_edgetpu();
    interpreter_built = build_interpreter();
    interpreter_configured = configure_interpreter();
  }

  MobNetBackbone_t(std::string model_filename, std::string device_path) : model_filename(model_filename) {
    
    auto tpus = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
    for (auto& tpu : tpus) {
      spdlog::info("TPU Devices: {}", tpu.path);
    }

    model_loaded = load_model();
    edgetpu_connected = connect_edgetpu(edgetpu::DeviceType::kApexUsb, device_path);
    interpreter_built = build_interpreter();
    interpreter_configured = configure_interpreter();

    auto* input_tensor = interpreter->input_tensor(0);
    model_width = input_tensor->dims->data[1];
    model_height = input_tensor->dims->data[2];
    model_channels = input_tensor->dims->data[3];
  }

  auto valid() -> bool { return model_loaded && edgetpu_connected && interpreter_built && interpreter_configured; }

  auto log_details() -> void {
    spdlog::info("MobNet Backbone");
    spdlog::info("\tmodel_loaded: {}", model_loaded);
    spdlog::info("\tedgetpu_connected: {}", edgetpu_connected);
    spdlog::info("\tinterpreter_built: {}", interpreter_built);
    spdlog::info("\tinterpreter_configured: {}", interpreter_configured);

    if (interpreter == nullptr) return;

    spdlog::info("\t-----------------------");
    spdlog::info("\tmodel: {}", model_filename);
    spdlog::info("\ttensors.size: {}", interpreter->tensors_size());
    spdlog::info("\tnodes.size: {}", interpreter->nodes_size());

    for (size_t input_index = 0; input_index < interpreter->inputs().size(); ++input_index) {
      auto input_tensor = interpreter->input_tensor(input_index);
      spdlog::info("\tinput_tensor[{}].name: {}", input_index, input_tensor->name);
      spdlog::info("\tinput_tensor[{}].bytes: {}", input_index, input_tensor->bytes);
      // spdlog::info("\tinput_tensor[{}].type: {}", input_index, input_tensor->type);
    }

    for (size_t output_index = 0; output_index < interpreter->outputs().size(); ++output_index) {
      auto output_tensor = interpreter->output_tensor(output_index);
      spdlog::info("\toutput_tensor[{}].name: {}", output_index, output_tensor->name);
      spdlog::info("\toutput_tensor[{}].bytes: {}", output_index, output_tensor->bytes);
      // spdlog::info("\toutput_tensor[{}].type: {}", output_index, output_tensor->type);
    }

    spdlog::info("\tdebug_logging: {}", debug_logging);
  }

  auto log_times() -> void {
    if (interpreter == nullptr) return;

    spdlog::info("object_detector_times");
    spdlog::info("\tcopy_times: {}", copy_times);
    spdlog::info("\tinfer_times: {}", infer_times);
    spdlog::info("\tprocess_times: {}", process_times);
  }

};

std::vector<std::vector<std::vector<std::vector<int8_t>>>> GetbboxFeatures (std::vector<std::vector<std::vector<std::vector<int8_t>>>> features,std::vector<float> Bbox, int scaling){
    int xmin_scaled = Bbox[0]/scaling;
    int ymin_scaled = Bbox[1]/scaling;
    int xmax_scaled = Bbox[2]/scaling;
    int ymax_scaled = Bbox[3]/scaling;
    std::vector<std::vector<std::vector<std::vector<int8_t>>>> outfeatures;
    int dim0 = features.size();
    int dim3 = features [0][0][0].size();
    for (int i=0; i<dim0; i++){
        std::vector<std::vector<std::vector<int8_t>>> temp0;
        for (int j=ymin_scaled; j<=ymax_scaled; j++){
            std::vector<std::vector<int8_t>> temp1;
            for (int k=xmin_scaled; k<=xmax_scaled; k++){
                std::vector<int8_t> temp2;
                for (int l=0; l<dim3; l++){
                    temp2.push_back(features[i][j][k][l]);
                }
                temp1.push_back(temp2);
            }
            temp0.push_back(temp1);
        }
        outfeatures.push_back(temp0);
    }
    return outfeatures;
}

std::vector<std::vector<std::vector<std::vector<int8_t>>>> resizing_bilinear (std::vector<std::vector<std::vector<std::vector<int8_t>>>> features, int model_width, int model_height){
    // will resize the vector from 1,?,?,1792 to 1,10,10,1792 (based on bilinear interpolation)
    float dim0 = features.size();
    float dim1 = features[0].size();
    float dim2 = features[0][0].size();
    float dim3 = features[0][0][0].size();

    std::vector<std::vector<std::vector<std::vector<int8_t>>>> features_resized(
        dim0, std::vector<std::vector<std::vector<int8_t>>>(
            model_height, std::vector<std::vector<int8_t>>(
                model_width, std::vector<int8_t>(
                    dim3, 0))));

    float h_scale_factor = dim1/model_height;
    float w_scale_factor = dim2/model_width;
    int8_t q;
    int8_t q1;
    int8_t q2;
    int8_t v1;
    int8_t v2;
    int8_t v3;
    int8_t v4;
    float x;
    float y;
    float x_floor;
    float y_floor;
    float x_ceil;
    float y_ceil;

    for (float i=0; i<dim0;i++){
        for (float j=0; j<model_height;j++){
            for (float k=0; k<model_width;k++){
                for (int l=0; l<dim3;l++){
                    x = j*h_scale_factor;
                    y = k*w_scale_factor;
                    x_floor = std::floor(x);
                    y_floor = std::floor(y);
                    x_ceil = std::min(dim1-1, std::ceil(x));
                    y_ceil = std::min(dim2-1, std::ceil(y));
                    if (x_ceil==x_floor && y_ceil == y_floor){
                        q = features[i][x][y][l];
                    }
                    else if (x_ceil == x_floor){
                        q1 = features[i][x][y_floor][l];
                        q2 = features[i][x][y_ceil][l];
                        q = (q1 * (y_ceil - y)) + (q2 * (y - y_floor));
                    }
                    else if(y_ceil == y_floor){
                        q1 = features[i][x_floor][y][l];
                        q2 = features[i][x_ceil][y][l];
                        q = (q1 * (x_ceil - x)) + (q2	 * (x - x_floor));
                    }
                    else {
                        v1 = features[i][x_floor][y_floor][l];
                        v2 = features[i][x_ceil][y_floor][l];
                        v3 = features[i][x_floor][y_ceil][l];
                        v4 = features[i][x_ceil][y_ceil][l];
                        q1 = v1*(x_ceil-x) + v2 * (x-x_floor);
                        q2 = v3*(x_ceil-x) + v4 * (x-x_floor);
                        q = q1 * (y_ceil - y) + q2 * (y - y_floor);
                    }
                    features_resized[i][j][k][l] = q;
                }

            }
        }
    }
    return features_resized;
}

long double CosineSimilarity(std::vector<float> vector1, std::vector<float> vector2){
    long double cosine_similarity;
    long double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    for(int i = 0; i < vector1.size(); i++){
        dot += vector1[i] * vector2[i];
        denom_a += vector1[i] * vector1[i] ;
        denom_b += vector2[i] * vector2[i] ;
    }
    cosine_similarity = dot / (sqrt(denom_a) * sqrt(denom_b));
    return 1-cosine_similarity;
}

std::vector<int8_t> flatten(const std::vector<std::vector<std::vector<std::vector<int8_t>>>>& nestedVector) {
    std::vector<int8_t> flattenedVector;
    for (const auto& vec1 : nestedVector) {
        for (const auto& vec2 : vec1) {
            for (const auto& vec3 : vec2) {
                for (int8_t value : vec3) {
                    flattenedVector.push_back(value);
                }
            }
        }
    }

    return flattenedVector;
}



std::vector<int> Matching() {

    // set them true if showing results on the images are needed. DEBUG_SHOW_EPI_EMB also shows the eppipolar lines on the images with the costs
    bool DEBUG_SHOW = true;
    bool DEBUG_SHOW_EPI_EMB = true;
    // ALPHA is to weigh the epipolar and embedding costs; as ALPHA*epipolar_cost and (1-ALPHA)*embedding_cost
    float ALPHA= 0.5;
    // The following parameter is the scaling factor to get corresponding features from the feature map. Based on the current model, this should be 8
    int SCALING_FACTOR_BBOX_FEATURE_EXTRACTION = 8;



    // Loading the embedding model, which gives an embedding vector with size 256 for each bounding box that will be used to get the similarity cost
    std::string device_name{"/sys/bus/usb/devices/2-1"};
    std::string model_embedding_filename{"/opt/imagr/models/EmbeddingModel_MobNet_FeatureBased2_edgetpu.tflite"};
    MobNetBackbone_t embedding(model_embedding_filename, device_name);


    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //// This part consists of what Matching function expects as inputs, and should be obtained from the object detector model///
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Getting camera image views (only needed to show the results)
    std::string IMG_PATH_0{ "/opt/imagr/etc/test_data/shah_test/0.rgb" };
    std::string IMG_PATH_1{ "/opt/imagr/etc/test_data/shah_test/1.rgb" };
    std::string IMG_PATH_2{ "/opt/imagr/etc/test_data/shah_test/2.rgb" };
    cv::Mat IMG_0 = ReadRGB(IMG_PATH_0);
    cv::Mat IMG_1 = ReadRGB(IMG_PATH_1);
    cv::Mat IMG_2 = ReadRGB(IMG_PATH_2);
    // Getting Bounding box data for each view: xmin, ymin, xmax, ymax
    // Bbox# is a vector of vectors, each representing a bounding box. Some noises have also been defined. Correct matches are the first bounding boxes
    std::vector<std::vector<float>> Bbox0{{0.45646057 * 320, 0.46136874 * 320, 0.7116858 * 320, 0.68714494 * 320},{0.2 * 320, 0.1 * 320, 0.3 * 320,0.3 * 320},{0.15 * 320,0.4 * 320,0.35 * 320,0.6 * 320}};
    std::vector<std::vector<float>> Bbox1{{0.3533888 * 320, 0.0220868 * 320, 0.716594 * 320, 0.468731 * 320},{0.15 * 320,0.4 * 320,0.35 * 320,0.6 * 320},{0.2 * 320, 0.1 * 320, 0.3 * 320,0.3 * 320}};
    std::vector<std::vector<float>> Bbox2{{0.13252082 * 320, 0.004908189 * 320, 0.5448078 * 320, 0.36811334 * 320},{0.5 * 320,0.4 * 320,0.7 * 320,0.6 * 320},{0.1 * 320,0.8 * 320,0.2 * 320,0.9 * 320}  };
    // Getting corresponding features as examples
    // These features should be obtained from the object detector model (based on Mobnet backbone)
    // feature# is 1*40*40*1792 (obtained from concatentaing three output layers of the object detector model)
    const std::string json_filename_test_features= "/opt/imagr/etc/test_data/shah_test/test_features.json";
    std::ifstream f_test(json_filename_test_features);
    auto test_features = nlohmann::json::parse(f_test);
    std::vector<std::vector<std::vector<std::vector<int8_t>>>> feature0 = test_features["0"].get<std::vector<std::vector<std::vector<std::vector<int8_t>>>>>();
    std::vector<std::vector<std::vector<std::vector<int8_t>>>> feature1 = test_features["1"].get<std::vector<std::vector<std::vector<std::vector<int8_t>>>>>();
    std::vector<std::vector<std::vector<std::vector<int8_t>>>> feature2 = test_features["2"].get<std::vector<std::vector<std::vector<std::vector<int8_t>>>>>();
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    if (embedding.edgetpu_connected) {
        // Showing all the bounding boxes and save the results if DEBUG_SHOW is true
        if (DEBUG_SHOW == true) {
            cv::Mat Unmatched = ShowBBoxAll(IMG_0.clone(), IMG_1.clone(), IMG_2.clone(), Bbox0, Bbox1, Bbox2);
            cv::imwrite("Unmatched.png",Unmatched);
        }
        
        // Getting all the possible combinations of triple bounding boxes in all three views.
        // In Combinations, (i,j,k) encodes the ith bounding box in view0, jth bounding box in view1, kth bounding box in view2
        int num_Bbox0 = Bbox0.size();
        int num_Bbox1 = Bbox1.size();
        int num_Bbox2 = Bbox2.size();
        std::vector<std::vector<int>> Combinations;
        for (int i = 0; i < num_Bbox0;i++) {
            for (int j = 0; j < num_Bbox1;j++) {
                for (int k = 0; k < num_Bbox2;k++)
                    Combinations.push_back({ i,j,k });
            }
        }
        int num_combinations = Combinations.size();


        // two vectors to store the total epipolar costs and similarity costs
        std::vector<float> EpiCostTotal;
        std::vector<float> EmbCostTotal;

        // This part goes through the all possible combinations and computes both eppipolar and similarity costs
        for (int i = 0; i < num_combinations;i++) {
            // TripleBbox is vector of three indices showing the bounding box number in each view
            std::vector<int> TripleBbox = Combinations[i];

            ///////////////////////////////////////
            //// Eppipolar cost computation part///
            ///////////////////////////////////////

            Eigen::VectorXf Coeff01(3);
            Eigen::VectorXf Coeff02(3);
            Coeff01 = ComputeEpipolarLine(Calib, Bbox0[TripleBbox[0]], "cam0", "cam1");
            Coeff02 = ComputeEpipolarLine(Calib, Bbox0[TripleBbox[0]], "cam0", "cam2");
            float EpiDistance01 = GetLineDistance(Coeff01, Bbox1[TripleBbox[1]]);
            float EpiDistance02 = GetLineDistance(Coeff02, Bbox2[TripleBbox[2]]);

            Eigen::VectorXf Coeff10(3);
            Eigen::VectorXf Coeff12(3);
            Coeff10 = ComputeEpipolarLine(Calib, Bbox1[TripleBbox[1]], "cam1", "cam0");
            Coeff12 = ComputeEpipolarLine(Calib, Bbox1[TripleBbox[1]], "cam1", "cam2");
            float EpiDistance10 = GetLineDistance(Coeff10, Bbox0[TripleBbox[0]]);
            float EpiDistance12 = GetLineDistance(Coeff12, Bbox2[TripleBbox[2]]);

            Eigen::VectorXf Coeff20(3);
            Eigen::VectorXf Coeff21(3);
            Coeff20 = ComputeEpipolarLine(Calib, Bbox2[TripleBbox[2]], "cam2", "cam0");
            Coeff21 = ComputeEpipolarLine(Calib, Bbox2[TripleBbox[2]], "cam2", "cam1");
            float EpiDistance20 = GetLineDistance(Coeff20, Bbox0[TripleBbox[0]]);
            float EpiDistance21 = GetLineDistance(Coeff21, Bbox1[TripleBbox[1]]);

            float EpiCost = (((EpiDistance01 + EpiDistance10) / 2) + ((EpiDistance02 + EpiDistance20) / 2) + ((EpiDistance12 + EpiDistance21) / 2)) / 3;
            EpiCostTotal.push_back(EpiCost);
            ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


            //////////////////////////////////////////////////
            //// Embedding/Similarity cost computation part///
            /////////////////////////////////////////////////

            std::vector<std::vector<std::vector<std::vector<int8_t>>>> bbox0_features = GetbboxFeatures (feature0,Bbox0[TripleBbox[0]], SCALING_FACTOR_BBOX_FEATURE_EXTRACTION);
            std::vector<std::vector<std::vector<std::vector<int8_t>>>> bbox0_features_resized = resizing_bilinear (bbox0_features, embedding.model_width, embedding.model_height);

            std::vector<std::vector<std::vector<std::vector<int8_t>>>> bbox1_features = GetbboxFeatures (feature1,Bbox1[TripleBbox[1]], SCALING_FACTOR_BBOX_FEATURE_EXTRACTION);
            std::vector<std::vector<std::vector<std::vector<int8_t>>>> bbox1_features_resized = resizing_bilinear (bbox1_features, embedding.model_width, embedding.model_height);

            std::vector<std::vector<std::vector<std::vector<int8_t>>>> bbox2_features = GetbboxFeatures (feature2,Bbox2[TripleBbox[2]], SCALING_FACTOR_BBOX_FEATURE_EXTRACTION);
            std::vector<std::vector<std::vector<std::vector<int8_t>>>> bbox2_features_resized = resizing_bilinear (bbox2_features, embedding.model_width, embedding.model_height);

            std::vector<int8_t>  bbox0_features_resized_flattened = flatten(bbox0_features_resized);
            std::vector<int8_t>  bbox1_features_resized_flattened = flatten(bbox1_features_resized);
            std::vector<int8_t>  bbox2_features_resized_flattened = flatten(bbox2_features_resized);

            auto* input_tensor = embedding.interpreter->input_tensor(0);
            float scale_input = input_tensor ->params.scale;
            int32_t zero_point_input = input_tensor ->params.zero_point;

            auto* output_tensor = embedding.interpreter->output_tensor(0);
            float scale_output = output_tensor ->params.scale;
            int32_t zero_point_output = output_tensor ->params.zero_point;
            int output_size = output_tensor->dims->data[1];

            //Getting feature embeddings for each bounding box; feature_embedding_#_float vectors are the outputs that will be used for similarity measurements
            //CAM-0------------------------------------------------------------------
            std::memcpy(tflite::GetTensorData<int8_t>(input_tensor), bbox0_features_resized_flattened.data(),  bbox0_features_resized_flattened.size()*sizeof(int8_t));
            embedding.interpreter->Invoke();
            int8_t* feature_embedding_0 = tflite::GetTensorData<int8_t>(embedding.interpreter->output_tensor(0));
            std::vector<int8_t> feature_embedding_0_data(output_size, 0);
            std::memcpy(feature_embedding_0_data.data(), feature_embedding_0, feature_embedding_0_data.size());
            feature_embedding_0 = feature_embedding_0_data.data();
            std::vector<float> feature_embedding_0_float(output_size, 0);
            for (int i = 0; i < output_size;i++) {
                feature_embedding_0_float[i] = (feature_embedding_0[i]/scale_output)-zero_point_output;
            }
            //--------------------------------------------------------------------


            //CAM-1----------------------------------------------------------------
            std::memcpy(tflite::GetTensorData<int8_t>(input_tensor), bbox1_features_resized_flattened.data(),  bbox1_features_resized_flattened.size()*sizeof(int8_t));
            embedding.interpreter->Invoke();
            int8_t* feature_embedding_1 = tflite::GetTensorData<int8_t>(embedding.interpreter->output_tensor(0));
            std::vector<int8_t> feature_embedding_1_data(output_size, 0);
            std::memcpy(feature_embedding_1_data.data(), feature_embedding_1, feature_embedding_1_data.size());
            feature_embedding_1 = feature_embedding_1_data.data();
            std::vector<float> feature_embedding_1_float(output_size, 0);
            for (int i = 0; i < output_size;i++) {
                feature_embedding_1_float[i] = (feature_embedding_1[i]/scale_output)-zero_point_output;
            }
            //--------------------------------------------------------------------


            //CAM-2----------------------------------------------------------------
            std::memcpy(tflite::GetTensorData<int8_t>(input_tensor), bbox2_features_resized_flattened.data(),  bbox2_features_resized_flattened.size()*sizeof(int8_t));
            embedding.interpreter->Invoke();
            int8_t* feature_embedding_2 = tflite::GetTensorData<int8_t>(embedding.interpreter->output_tensor(0));
            std::vector<int8_t> feature_embedding_2_data(output_size, 0);
            std::memcpy(feature_embedding_2_data.data(), feature_embedding_2, feature_embedding_2_data.size());
            feature_embedding_2 = feature_embedding_2_data.data();
            std::vector<float> feature_embedding_2_float(output_size, 0);
            for (int i = 0; i < output_size;i++) {
                feature_embedding_2_float[i] = (feature_embedding_2[i]/scale_output)-zero_point_output;
            }
            
            //--------------------------------------------------------------------
            float EmbCost = (CosineSimilarity(feature_embedding_0_float,feature_embedding_1_float)+CosineSimilarity(feature_embedding_0_float,feature_embedding_2_float)+CosineSimilarity(feature_embedding_1_float,feature_embedding_2_float))/3;
            EmbCostTotal.push_back(EmbCost);
        }

        std::vector<float> NormalizedEpiCostTotal;
        std::vector<float> TotalCost;
        
        
        float min_epi = *std::min_element(EpiCostTotal.begin(), EpiCostTotal.end());
        float max_epi = *std::max_element(EpiCostTotal.begin(), EpiCostTotal.end());
        for (int i = 0; i < EpiCostTotal.size();i++){
            NormalizedEpiCostTotal.push_back(((EpiCostTotal[i] - min_epi) / (max_epi - min_epi)));
            TotalCost.push_back((ALPHA*NormalizedEpiCostTotal[i])+((1-ALPHA)*EmbCostTotal[i]));
            // spdlog::info("TotalCost[i]: {}", TotalCost[i]);
        }
        
        // Show the eppipolar lines on the images and save the image if DEBUG_SHOW_EPI_EMB is true
        if (DEBUG_SHOW_EPI_EMB == true) {
            for (int i = 0; i < num_combinations;i++) {
                std::vector<int> TripleBbox = Combinations[i];
                cv::Mat EpiImg = ShowEppipolars(IMG_0.clone(), IMG_1.clone(), IMG_2.clone(), Calib, Bbox0[TripleBbox[0]], Bbox1[TripleBbox[1]], Bbox2[TripleBbox[2]], EpiCostTotal[i],EmbCostTotal[i],TotalCost[i]);
                cv::imwrite(std::to_string(i)+"_epi.png",EpiImg);
            }
        }

        // Finding the minimum cost set of triple bounding boxes
        float MinCost = 10000.0;
        int MinIndex = 1000;
        for (int i = 0; i < TotalCost.size();i++) {
            if (TotalCost[i] <= MinCost) {
                MinCost = TotalCost[i];
                MinIndex = i;
            }
        }

        // Showing the matching bounding boxes
        if (DEBUG_SHOW == true) {
            cv::Scalar color(0, 255, 0);
            cv::Mat IMG0Matched = DrawBbox(IMG_0.clone(), Bbox0[Combinations[MinIndex][0]], color);
            cv::Mat IMG1Matched = DrawBbox(IMG_1.clone(), Bbox1[Combinations[MinIndex][1]], color);
            cv::Mat IMG2Matched = DrawBbox(IMG_2.clone(), Bbox2[Combinations[MinIndex][2]], color);
            cv::Mat concatenated1;
            cv::hconcat(IMG0Matched, IMG1Matched, concatenated1);
            cv::Mat concatenated2;
            cv::hconcat(concatenated1, IMG2Matched, concatenated2);
            cv::imwrite("matched.png",concatenated2);
            //cv::imshow("Epipolar Lines", concatenated2);
            //int k = cv::waitKey(0);
        }
    std::vector<int> RESULTS = Combinations[MinIndex];
    return RESULTS;
    }

    else{
        spdlog::info("Edg-TPU is not connected; the program cannot proceed");
        std::vector<int> vec{-1,-1,-1};
        return vec;
    }
    
}


TEST_CASE("BoundingBoxMatching","[matching]") {
    std::vector<int> RESULTS = Matching();
    spdlog::info("Matching Bounding Boxes: ");
    spdlog::info("Bounding Box Number/ID in Cam0: {}", RESULTS[0]);
    spdlog::info("Bounding Box Number/ID in Cam1: {}", RESULTS[1]);
    spdlog::info("Bounding Box Number/ID in Cam2: {}", RESULTS[2]);

}
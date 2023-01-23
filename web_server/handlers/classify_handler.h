#ifndef PERSONHANDLER_H
#define PERSONHANDLER_H

#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTMLForm.h"
#include "Poco/Net/PartHandler.h"
#include "Poco/Net/MessageHeader.h"
#include "Poco/Net/ServerSocket.h"
#include "Poco/Timestamp.h"
#include "Poco/DateTimeFormatter.h"
#include "Poco/DateTimeFormat.h"
#include "Poco/Exception.h"
#include "Poco/ThreadPool.h"
#include "Poco/Util/ServerApplication.h"
#include "Poco/Util/Option.h"
#include "Poco/Util/OptionSet.h"
#include "Poco/Util/HelpFormatter.h"
#include "Poco/StreamCopier.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include "../model_meta.hpp"

#include <onnxruntime_cxx_api.h>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using Poco::DateTimeFormat;
using Poco::DateTimeFormatter;
using Poco::ThreadPool;
using Poco::Timestamp;
using Poco::Net::HTMLForm;
using Poco::Net::HTTPRequestHandler;
using Poco::Net::HTTPRequestHandlerFactory;
using Poco::Net::HTTPServer;
using Poco::Net::HTTPServerParams;
using Poco::Net::HTTPServerRequest;
using Poco::Net::HTTPServerResponse;
using Poco::Net::NameValueCollection;
using Poco::Net::ServerSocket;
using Poco::Util::Application;
using Poco::Util::HelpFormatter;
using Poco::Util::Option;
using Poco::Util::OptionCallback;
using Poco::Util::OptionSet;
using Poco::Util::ServerApplication;



class MyPartHandler : public Poco::Net::PartHandler
{
public:
    MyPartHandler(std::vector<int64_t>* input_dims) : _input_dims(input_dims) {};
    cv::Mat _img;

    void handlePart([[maybe_unused]] const Poco::Net::MessageHeader& header, std::istream& stream)
    {
        // from stream to string
        std::string img_binary;
        Poco::StreamCopier::copyToString(stream, img_binary);

        // to bytes
        std::vector<std::byte> bytes;
        bytes.reserve(img_binary.size());
        std::transform(img_binary.begin(), img_binary.end(), std::back_inserter(bytes), [](char c){
            return std::byte(c);
        });

        // to image
        cv::Mat cv_bytes_array(1, bytes.size(), CV_8UC1, bytes.data());
        cv::Mat img = cv::imdecode(cv_bytes_array, cv::IMREAD_UNCHANGED);

        // preprocess image
        cv::Mat resized_bgr;
        cv::resize(
            img,
            resized_bgr,
            cv::Size(_input_dims->at(2), _input_dims->at(3)),
            cv::InterpolationFlags::INTER_CUBIC);

        cv::Mat resized_rgb;
        cv::cvtColor(resized_bgr, resized_rgb, cv::ColorConversionCodes::COLOR_BGR2RGB);

        cv::Mat resized;
        resized_rgb.convertTo(resized, CV_32F, 1.0 / 255);

        cv::Mat channels[3];
        cv::split(resized, channels);
        channels[0] = (channels[0] - 0.485) / 0.229;
        channels[1] = (channels[1] - 0.456) / 0.224;
        channels[2] = (channels[2] - 0.406) / 0.225;
        cv::merge(channels, 3, resized);

        cv::dnn::blobFromImage(resized, _img);
    }
private:
    std::vector<int64_t>* _input_dims;
};



template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}



class ClassifyHandler : public HTTPRequestHandler
{
public:
    ClassifyHandler(const std::string &format, ModelMeta model_meta):
        _format(format),
        _model_meta(model_meta)
    {
        _ph = new MyPartHandler(model_meta.input_dims);
    }

    void handleRequest(HTTPServerRequest &request, HTTPServerResponse &response)
    {
        HTMLForm form(request, request.stream(), *_ph);
        response.setChunkedTransferEncoding(true);
        response.setContentType("application/json");
        std::ostream &ostr = response.send();

        try
        {
            // prepare model input
            size_t input_tensor_size = vectorProduct(*_model_meta.input_dims);
            std::vector<float> input_tensor_values(input_tensor_size);
            std::copy(_ph->_img.begin<float>(), _ph->_img.end<float>(), input_tensor_values.begin());

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                *_model_meta.mem_info,
                input_tensor_values.data(),
                input_tensor_size,
                _model_meta.input_dims->data(),
                _model_meta.input_dims->size()));

            // prepare model output
            std::vector<Ort::Value> output_tensors;
            size_t output_tensor_size = vectorProduct(*_model_meta.output_dims);
            std::vector<float> output_tensor_values(output_tensor_size);
            output_tensors.push_back(Ort::Value::CreateTensor<float>(
                *_model_meta.mem_info,
                output_tensor_values.data(),
                output_tensor_size,
                _model_meta.output_dims->data(),
                _model_meta.output_dims->size()));

            // run model
            _model_meta.sess->Run(
                Ort::RunOptions{nullptr},
                _model_meta.input_names->data(),
                input_tensors.data(),
                1,
                _model_meta.output_names->data(),
                output_tensors.data(),
                1);

            // interpret predictionn
            int pred_id = 0;
            float activation = 0;
            float max_activation = std::numeric_limits<float>::lowest();
            for (size_t i = 0; i < _model_meta.labels->size(); i++) {
                activation = output_tensor_values.at(i);
                if (activation > max_activation) {
                    pred_id = i;
                    max_activation = activation;
                }
            }

            std::string pred_label = _model_meta.labels->at(pred_id);
            ostr << "{\"label_id\": " << pred_id << ", \"label\": \"" << pred_label << "\"}";
            return;
        }
        catch (std::exception &e)
        {
            ostr << "{\"result\": false, \"reason\": \"" << e.what() << "\"}";
            return;
        }
    }

private:
    std::string _format;
    ModelMeta _model_meta;
    MyPartHandler* _ph;
};
#endif // !PERSONHANDLER_H

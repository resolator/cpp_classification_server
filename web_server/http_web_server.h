#ifndef HTTPWEBSERVER_H
#define HTTPWEBSERVER_H

#include "Poco/Net/HTTPServer.h"
#include "Poco/Net/HTTPRequestHandler.h"
#include "Poco/Net/HTTPRequestHandlerFactory.h"
#include "Poco/Net/HTTPServerParams.h"
#include "Poco/Net/HTTPServerRequest.h"
#include "Poco/Net/HTTPServerResponse.h"
#include "Poco/Net/HTTPServerParams.h"
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
#include "model_meta.hpp"

using Poco::DateTimeFormat;
using Poco::DateTimeFormatter;
using Poco::ThreadPool;
using Poco::Timestamp;
using Poco::Net::HTTPRequestHandler;
using Poco::Net::HTTPRequestHandlerFactory;
using Poco::Net::HTTPServer;
using Poco::Net::HTTPServerParams;
using Poco::Net::HTTPServerRequest;
using Poco::Net::HTTPServerResponse;
using Poco::Net::ServerSocket;
using Poco::Util::Application;
using Poco::Util::HelpFormatter;
using Poco::Util::Option;
using Poco::Util::OptionCallback;
using Poco::Util::OptionSet;
using Poco::Util::ServerApplication;

#include "http_request_factory.h"



class HTTPWebServer : public Poco::Util::ServerApplication
{
public:
    HTTPWebServer() : _helpRequested(false) {}
    ~HTTPWebServer(){}

protected:
    void initialize(Application &self)
    {
        loadConfiguration();
        ServerApplication::initialize(self);
    }



    void uninitialize()
    {
        ServerApplication::uninitialize();
    }



    void defineOptions(OptionSet &options)
    {
        ServerApplication::defineOptions(options);

        options.addOption(
            Option("help", "h", "Display argument help information.")
                .required(false)
                .repeatable(false)
                .callback(OptionCallback<HTTPWebServer>(this, &HTTPWebServer::handleHelp)));
        options.addOption(
            Option("model-path", "m", "Path to ONNX model.")
                .required(true)
                .repeatable(false)
                .argument("value")
                .callback(OptionCallback<HTTPWebServer>(this, &HTTPWebServer::handleModelPath)));
        options.addOption(
            Option("labels-path", "m", "Path to file with labels for the model.")
                .required(true)
                .repeatable(false)
                .argument("value")
                .callback(OptionCallback<HTTPWebServer>(this, &HTTPWebServer::handleLabelsPath)));
    }



    void handleHelp([[maybe_unused]] const std::string &name,
                    [[maybe_unused]] const std::string &value)
    {
        HelpFormatter helpFormatter(options());
        helpFormatter.setCommand(commandName());
        helpFormatter.setUsage("OPTIONS");
        helpFormatter.setHeader("A web server with ResNet18 classification model.");
        helpFormatter.format(std::cout);
        stopOptionsProcessing();
        _helpRequested = true;
    }



    void handleModelPath([[maybe_unused]] const std::string &name,
                         [[maybe_unused]] const std::string &value)
    {
        _model_path = value;
    }



    void handleLabelsPath([[maybe_unused]] const std::string &name,
                          [[maybe_unused]] const std::string &value)
    {
        _labels_path = value;
    }



    std::vector<std::string> readLabels(std::string& labels_path)
    {
        std::vector<std::string> labels;
        std::string line;
        std::ifstream fp(labels_path);

        while (std::getline(fp, line)) {
            labels.push_back(line);
        }
        return labels;
    }



    int main([[maybe_unused]] const std::vector<std::string> &args)
    {
        if (!_helpRequested) {
            // init model
            std::string instance_name{"image-classification-inference"};

            Ort::Env env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, instance_name.c_str());

            Ort::SessionOptions sess_opts;
            sess_opts.SetIntraOpNumThreads(1);
            sess_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            Ort::Session sess(env, _model_path.c_str(), sess_opts);
            Ort::AllocatorWithDefaultOptions allocator;

            // init model input
            const char* input_name = sess.GetInputName(0, allocator);
            Ort::TypeInfo input_type_info = sess.GetInputTypeInfo(0);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> input_dims = input_tensor_info.GetShape();
            if (input_dims.at(0) == -1) {
                input_dims.at(0) = 1;
            }

            // init model output
            const char* output_name = sess.GetOutputName(0, allocator);
            Ort::TypeInfo output_type_info = sess.GetOutputTypeInfo(0);
            auto outputTensorInfo = output_type_info.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_dims = outputTensorInfo.GetShape();
            if (output_dims.at(0) == -1) {
                output_dims.at(0) = 1;
            }

            // init model meta
            Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

            std::vector<const char*> input_names{input_name};
            std::vector<const char*> output_names{output_name};

            std::vector<std::string> labels{readLabels(_labels_path)};

            ModelMeta model_meta;
            model_meta.input_dims = &input_dims;
            model_meta.output_dims = &output_dims;
            model_meta.mem_info = &mem_info;
            model_meta.sess = &sess;
            model_meta.input_names = &input_names;
            model_meta.output_names = &output_names;
            model_meta.labels = &labels;

            // init server
            unsigned short port = (unsigned short)config().getInt("HTTPWebServer.port", 80);
            std::string format(config().getString("HTTPWebServer.format", DateTimeFormat::SORTABLE_FORMAT));

            ServerSocket svs(Poco::Net::SocketAddress("0.0.0.0", port));
            HTTPServer srv(new HTTPRequestFactory(format, model_meta), svs, new HTTPServerParams);
            srv.start();
            waitForTerminationRequest();
            srv.stop();
        }
        return Application::EXIT_OK;
    }

private:
    bool _helpRequested;
    std::string _model_path;
    std::string _labels_path;

};
#endif // !HTTPWEBSERVER

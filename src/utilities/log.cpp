#include "src/utilities/log.h"

#include <chrono>
#include <ctime>

namespace openturbine::util {

Log* Log::log_instance_ = nullptr;

std::string SeverityLevelToString(const SeverityLevel& level) {
    switch (level) {
        case SeverityLevel::kDebug: {
            return std::string("DEBUG");
            break;
        }
        case SeverityLevel::kError: {
            return std::string("ERROR");
            break;
        }
        case SeverityLevel::kInfo: {
            return std::string("INFO");
            break;
        }
        case SeverityLevel::kWarning: {
            return std::string("WARNING");
            break;
        }
        default: {
            return std::string("NONE");
            break;
        }
    }
}

Log::Log(std::string name, SeverityLevel max_severity, OutputType type)
    : file_name_(name), max_severity_level_(max_severity), output_type_(type) {
}

Log* Log::Get(std::string name, SeverityLevel max_severity, OutputType type) {
    if (log_instance_ == nullptr) {
        log_instance_ = new Log(name, max_severity, type);
    }
    return log_instance_;
}

void Log::WriteMessage(const std::string& message, SeverityLevel severity) const {
    if (severity <= this->max_severity_level_) {
        // print the time stamp in YYYY-MM-DD HH:MM:SS format
        auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        auto local_t = std::localtime(&t);

        auto year = std::to_string(local_t->tm_year + 1900);
        auto month = std::to_string(local_t->tm_mon + 1);
        auto day = std::to_string(local_t->tm_mday);
        auto hr = std::to_string(local_t->tm_hour);
        auto min = std::to_string(local_t->tm_min);
        auto sec = std::to_string(local_t->tm_sec);
        auto time_stamp = year + std::string("-") + month + std::string("-") + day +
                          std::string(" ") + hr + std::string(":") + min + std::string(":") + sec;

        // print the annotated log message in the following format:
        // [time stamp] [openturbine] [severity] message
        auto log_message = std::string("[") + time_stamp + std::string("] ") +
                           std::string("[openturbine] [") + SeverityLevelToString(severity) +
                           std::string("] ") + message;

        std::ofstream out(this->file_name_, std::ofstream::out | std::ofstream::app);
        if (this->output_type_ == OutputType::kFile) {
            out << log_message;
            out.close();
            return;
        }

        if (this->output_type_ == OutputType::kConsole) {
            std::cout << log_message;
            return;
        }

        std::cout << log_message;
        out << log_message;
        out.close();
    }
}

}  // namespace openturbine::util

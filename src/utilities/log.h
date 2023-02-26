#pragma once

#include <fstream>
#include <iostream>
#include <string>

namespace openturbine::debug {

/// @brief Log severity levels - lower value indicates higher priority of incident
enum class SeverityLevel {
    kNone = 0,
    kError = 1,
    kWarning = 2,
    kInfo = 3,
    kDebug = 4
};

/// @brief Convert provided SeverityLevel -> to a string
std::string SeverityLevelToString(const SeverityLevel&);

/// @brief Convert provided string -> SeverityLevel value
SeverityLevel StringToSeverityLevel(const std::string&);

/// @brief Log output types
enum class OutputType {
    kConsole = 0,
    kFile = 1,
    kConsoleAndFile = 2
};

/// @brief Convert provided string -> OutputType value
OutputType StringToOutputType(const std::string&);

/*! \brief  A basic logging utility for OpenTurbine
 *
 *  A logging class based on the Singleton design pattern:
 *  https://en.wikipedia.org/wiki/Singleton_pattern.
 *  This class defines the `Get` method that serves as an alternative to a
 *  c-tor and lets clients access the same instance of this class.
 */
class Log {
public:
    virtual ~Log() = delete;
    Log(const Log&) = delete;
    Log& operator=(const Log&) = delete;
    Log(Log&&) = delete;
    Log& operator=(Log&&) = delete;

    /*!
     *  This is the static method that controls the access to the singleton
     *  instance of Log. On the first run, it creates a singleton object and
     *  places it into the static field. On subsequent runs, it returns the
     *  existing object stored in the static field.
     */
    static Log* Get(std::string name = "log.txt",
                    SeverityLevel max_severity = SeverityLevel::kDebug,
                    OutputType type = OutputType::kConsoleAndFile);

    std::string GetOutputFileName() const { return file_name_; }
    SeverityLevel GetMaxSeverityLevel() const { return max_severity_level_; }
    OutputType GetOutputType() const { return output_type_; }

    /// @brief Write a logging message using the Log object
    /// @param SeverityLevel: Indicates the severity level of the log message
    void WriteMessage(const std::string&, SeverityLevel) const;

    void Debug(std::string message) const { WriteMessage(message, SeverityLevel::kDebug); }
    void Error(std::string message) const { WriteMessage(message, SeverityLevel::kError); }
    void Info(std::string message) const { WriteMessage(message, SeverityLevel::kInfo); }
    void Warning(std::string message) const { WriteMessage(message, SeverityLevel::kWarning); }

private:
    std::string file_name_;
    SeverityLevel max_severity_level_;  //!< max severity level for logging
    OutputType output_type_;

    static Log* log_instance_;  //!< static instance of the ptr to Log

    /*!
     *  A private c-tor to prevent direct construction of Log class.
     *  The severity of the Log object indicates the max severity level of the logger -
     *  any message with a severity level higher than the max severity level will be
     *  printed. E.g. for max_severity_level_ = Info, None/Error/Warning messages will
     *  also be printed.
     */
    Log(std::string name = "log.txt", SeverityLevel max_severity = SeverityLevel::kDebug,
        OutputType type = OutputType::kConsoleAndFile);
};

}  // namespace openturbine::debug

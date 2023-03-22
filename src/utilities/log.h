#pragma once

#include <fstream>
#include <iostream>
#include <string>

namespace openturbine::util {

/*! @brief Log severity levels - this is a hierarchial list, lower numeric value
 *         indicates higher severity of incident
 */
enum class SeverityLevel {
    kNone = 0,
    kError = 1,
    kWarning = 2,
    kInfo = 3,
    kDebug = 4
};

/// @brief Convert provided SeverityLevel -> to a string
std::string SeverityLevelToString(const SeverityLevel&);

/// @brief Log output types
enum class OutputType {
    kConsole = 0,
    kFile = 1,
    kConsoleAndFile = 2
};

/*! \brief  A basic logging utility for OpenTurbine
 *
 *  A logging class based on the Singleton design pattern:
 *  https://en.wikipedia.org/wiki/Singleton_pattern.
 *  This class defines the `Get` method that serves as an alternative to a
 *  c-tor and lets clients access the same instance of this class.
 */
class Log {
public:
    ~Log() = default;
    /// Explicitly delete the copy c-tor
    Log(const Log&) = delete;
    /// Also delete the copy assignment c-tor
    Log& operator=(const Log&) = delete;
    Log(Log&&) = default;
    Log& operator=(Log&&) = default;

    /*!
     *  This is a static method that controls the access to the singleton
     *  instance of Log. On the first run, it creates a singleton object and
     *  places it into the static field. On subsequent runs, it returns the
     *  existing object stored in the static field.
     *
     *  @param name: Name of the log file, default is "log.txt"
     *  @param max_severity: Max severity level for logging, default is Debug.
     *      The severity of the Log object indicates the cutoff severity level of the logger -
     *      any log message with a severity level (numeric value) lower than the max severity
     *      level will be printed. E.g. for max severity level = Info -> None, Error, Warning
     *      and Info messages will be printed, but Debug messages will not.
     *  @param type: Output type for logging, default is Console and File
     *  @return The static instance of the ptr to Log
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
    SeverityLevel max_severity_level_;
    OutputType output_type_;
    static Log* log_instance_;

    /// A private c-tor to prevent direct construction of the Log class
    Log(std::string name = "log.txt", SeverityLevel max_severity = SeverityLevel::kDebug,
        OutputType type = OutputType::kConsoleAndFile);
};

}  // namespace openturbine::util

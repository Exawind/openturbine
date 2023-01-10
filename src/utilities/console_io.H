#ifndef CONSOLE_IO_H
#define CONSOLE_IO_H

#include <iostream>

namespace openturbine::io {

void print_banner(std::ostream& /*out*/);

void print_usage(std::ostream& /*out*/);

void print_error(const std::string& msg);

// void print_summary(std::ostream&);

void print_tpls(std::ostream& /*out*/);

} // namespace openturbine::io

#endif /* CONSOLE_IO_H */
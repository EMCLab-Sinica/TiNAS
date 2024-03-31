#include <stdarg.h>
#include <stdio.h>

void _DBGUART(char* format, ...) {
    va_list arg;
    va_start(arg, format);

    vprintf(format, arg);

    va_end(arg);
}

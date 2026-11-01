#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>
int main(int argc, char **argv) {
    int port = 0;
    if (argc == 2 && !strcmp(argv[1], "--version")) { puts("version: 0.8.0-dev (build 12030, commit abcdef123)"); return 0; }
    for (int i = 1; i + 1 < argc; ++i) if (!strcmp(argv[i], "--port")) port = atoi(argv[i+1]);
    int fd = socket(AF_INET, SOCK_STREAM, 0), yes = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    struct sockaddr_in addr = { .sin_family=AF_INET, .sin_port=htons(port), .sin_addr.s_addr=htonl(INADDR_LOOPBACK) };
    if (bind(fd, (void *)&addr, sizeof(addr)) || listen(fd, 5)) return 2;
    for (;;) {
        int c = accept(fd, NULL, NULL); if (c < 0) return 3;
        char buf[65536] = {0}; int n = read(c, buf, sizeof(buf)-1); if (n <= 0) { close(c); continue; }
        const char *body = "{\"content\":\"{\\\"answers\\\":{\\\"cal-0\\\":{\\\"choice\\\":\\\"a\\\",\\\"probabilities\\\":{\\\"a\\\":0.875,\\\"b\\\":0.125},\\\"confidence\\\":0.875}}}\"}";
        dprintf(c, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s", strlen(body), body);
        close(c);
    }
}

#include <stdlib.h>
#include <chrono>
#include <thread>
#include "./psstore.h"

int main(int argc, char *argv[]) {
  if (getenv("DMLC_NUM_WORKER") == NULL) {
    putenv("DMLC_NUM_WORKER=1");
  }
  if (getenv("DMLC_NUM_SERVER") == NULL) {
    putenv("DMLC_NUM_SERVER=1");
  }
  for (int i = 0; i< argc; i++) {
    if (!strcmp(argv[i], "worker")) {
      if (getenv("TENSORFLOW_ROLE") == NULL) {
        putenv("TENSORFLOW_ROLE=worker");
      }
    } else if (!strcmp(argv[i], "server")) {
      if (getenv("TENSORFLOW_ROLE") == NULL) {
        putenv("TENSORFLOW_ROLE=server");
      }
    } else if (!strcmp(argv[i], "scheduler")) {
      if (getenv("TENSORFLOW_ROLE") == NULL) {
        putenv("TENSORFLOW_ROLE=scheduler");
      }
    }
  }
  if (getenv("DMLC_PS_ROOT_URI") == NULL) {
    putenv("DMLC_PS_ROOT_URI=127.0.0.1");
  }
  if (getenv("DMLC_PS_ROOT_PORT") == NULL) {
    putenv("DMLC_PS_ROOT_PORT=22228");
  }
  tensorflow::psstore::PSStore *ps = tensorflow::psstore::PSStore::Create("dist_sync");
  if (ps != nullptr) {
    std::cout << "START Success" << std::endl;
  } else {
    std::cout << "START Fail" << std::endl;
  }
  //std::this_thread::sleep_for(std::chrono::seconds(3600));
  delete ps;
  ps = nullptr;
  return 0;
}

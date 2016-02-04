/*!
 * From dmlc/mxnet
 */

#include <stdlib.h>
#include "./psstore.h"
#include "./psstore_dist.h"
#include "./psstore_dist_server.h"

namespace tensorflow {
namespace psstore {
PSStore* PSStore::Create(const char *type_name) {
  std::string tname = type_name;
  std::transform(tname.begin(), tname.end(), tname.begin(), ::tolower);
  PSStore* ps = nullptr;
  if (tname == "dist_async" ||
      tname == "dist_sync" ||
      tname == "dist") {
    std::cout << "type_name: " << type_name << std::endl;
    ps = new PSStoreDist();
    ps->Run();
    if (tname == "dist_sync" &&
        ps->IsWorkerNode() &&
        ps->get_rank() == 0) {
      // configure the server to be the sync mode
      ps->SendCommandToServers(psstore::kSyncMode, "");
    }
  } else {
    std::cout << "Unknown PSStore type \"" << tname << "\"";
    return nullptr;
  }
  ps->type_ = tname;
  return ps;
}
}  // namespace psstore
}  // namespace tensorflow

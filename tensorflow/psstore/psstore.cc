/*!
 * From dmlc/mxnet
 */

#include <stdlib.h>
#include "./psstore.h"
#include "./psstore_dist.h"
#include "./psstore_dist_server.h"

namespace tensorflow {
namespace psstore {
PSStore* PSStore::Create(const char *type_name, const std::string& args) {
  std::string tname = type_name;
  std::transform(tname.begin(), tname.end(), tname.begin(), ::tolower);
  PSStore* ps = nullptr;
  if (tname == "dist_async" ||
      tname == "dist_sync" ||
      tname == "dist") {
    ps = new PSStoreDist();
    ps->Run();
    if (tname == "dist_sync" &&
        ps->IsWorkerNode() &&
        ps->get_rank() == 0) {
      // configure the server to be the sync mode
      ps->SendCommandToServers(psstore::kSyncMode, "");
      if (args.size()) {
        ps->SendCommandToServers(psstore::kInitUpdater, args);
      }
    }
  } else {
    VLOG(0) << "Unknown PSStore type \"" << tname << "\"";
    return nullptr;
  }
  ps->type_ = tname;
  return ps;
}

std::shared_ptr<PSStore> PSStore::_GetSharedRef(const char *type_name, const std::string& args) {
  static std::shared_ptr<PSStore> sptr(Create(type_name, args));
  return sptr;
}

PSStore* PSStore::Get(const char *type_name, const std::string& args) {
  static PSStore *inst = _GetSharedRef(type_name, args).get();
  return inst;
}
}  // namespace psstore
}  // namespace tensorflow

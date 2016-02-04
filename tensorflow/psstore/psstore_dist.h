/**
 * From dmlc/mxnet
 */
#ifndef TENSORFLOW_PS_PSSTORE_DIST_H_
#define TENSORFLOW_PS_PSSTORE_DIST_H_
#include <string>
#include <vector>
#include <unordered_map>
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "ps/ps.h"
#include "./psstore_dist_server.h"

namespace tensorflow {
namespace psstore {
/**
 * \brief distributed psstore
 *
 * for a worker node, it always guarantees that all push and pull issued from
 * this worker on the same key are serialized. namely push(3) and then pull(3),
 * then the data pulled is always containing the modification from the push(3).
 *
 * it's the server node's job to control the data consistency among all
 * workers. see details on \ref ServerHandle::Start
 */
class PSStoreDist : public PSStore {
 public:
  PSStoreDist() {
    if (IsWorkerNode()) {
      ps_worker_ = new ps::KVWorker<char>(0);
    } else if (IsServerNode()) {
      server_ = new PSStoreDistServer();
    }
  }

  ~PSStoreDist() {
    if (IsWorkerNode()) {
      if (get_rank() == 0) {
        // stop the executor at servers
        this->SendCommandToServers(kStopServer, "");
      }
      ps::Finalize();
      delete ps_worker_;
      ps_worker_ = nullptr;
    } else if (IsServerNode()) {
      ps::Finalize();
      delete server_;
      server_ = nullptr;
    } else if (IsSchedulerNode()) {
      ps::Finalize();
    }
  }

  void Init(const int key,
            const Tensor& value) override {
    if (get_rank() == 0) {
      Push(key, value, 0);
    } else {
      // do nothing
    }
    Barrier();
  }

  void Run() override {
    std::cout << "Begin running" << std::endl;
    if (IsWorkerNode()) {
      ps::Start("tensorflow_worker\0");
      std::cout << "I am worker." << std::endl;
    } else if (IsServerNode()) {
      ps::Start("tensorflow_server\0");
      std::cout << "I am server." << std::endl;
      if (server_) server_->Run();
    } else if (IsSchedulerNode()) {
      ps::Start("tensorflow_scheduler\0");
      std::cout << "I am scheduler." << std::endl;
    } else {
      std::cout << "Surprise!!!" << std::endl;
    }
  }

  void Push(const int key,
            const Tensor& value,
              int priority) override {
    TensorProto tp;
    value.AsProtoField(&tp);
    int size = tp.ByteSize();
    char* data = new char[size];
    tp.SerializeToArray(data, size);
    PSKV& pskv = EncodeKey(key, size);

    ps::SArray<char> vals(data, size, false);
    CHECK_NOTNULL(ps_worker_)->ZPush(
      pskv.keys, vals, pskv.lens, 0, nullptr);
    cache_[key] = size;
    delete data;
  }

  void Pull(const int key,
            Tensor* value,
            int priority) override {
    size_t size = cache_[key];
    // convert to ps keys
    PSKV& pskv = EncodeKey(key, size);
    // issue pull, false means no delete
    auto vals = new ps::SArray<char>(size);
    CHECK_NOTNULL(ps_worker_)->ZPull(
      pskv.keys, vals, &pskv.lens, 0, nullptr);
    TensorProto tp;
    tp.ParseFromArray(vals->data(), static_cast<int>(vals->size()));
    bool res = value->FromProto(tp);
    LOG(INFO) << res;
  }

  void set_updater(const Updater& updater) override {
    CHECK(updater) << "invalid updater";
    if (IsServerNode()) {
      CHECK_NOTNULL(server_)->set_updater(updater);
    } else {
      updater_ = updater;
    }
  }

  void set_controller(const Controller& controller) override {
    CHECK(controller) << "invalid controller";
    if (IsServerNode()) {
      CHECK_NOTNULL(server_)->set_controller(controller);
    } else {
      controller_ = controller;
    }
  }

  void Barrier() override {
    ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
  }

  void SendCommandToServers(int cmd_id,
                            const std::string& cmd_body) override {
    CHECK_NOTNULL(ps_worker_);
    ps_worker_->Wait(ps_worker_->Request(cmd_id, cmd_body, ps::kServerGroup));
  }

  int get_group_size() const override { return ps::NumWorkers(); }

  int get_rank() const override { return ps::MyRank(); }

 private:
  /**
   * \brief struct for ps keys and lens
   */
  struct PSKV {
    ps::SArray<ps::Key> keys;  // n keys
    ps::SArray<int> lens;  // the length of the i-th value
    int size;
  };

  /**
   * \brief cache all key partitions
   */
  std::unordered_map<int, PSKV> ps_kv_;

  /**
   * \brief serizelize EncodeKey
   */
  std::mutex mu_;

  /**
   * \brief convert to keys in ps
   */
  PSKV& EncodeKey(int key, size_t size) {
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();
    if (!pskv.keys.empty()) {
      CHECK_EQ(static_cast<size_t>(pskv.size), size) << "The value size cannot be changed";
    } else {
      auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
      int num_servers = krs.size();
      CHECK_GT(num_servers, 0);

      // a simple heuristic for load balance
      if (size < big_bound_) {
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        ps::Key ps_key = krs[server].begin() + key;
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(size);
        pskv.size = size;
      } else {
        // parition it to all servers
        pskv.size = 0;
        for (int i = 0; i < num_servers; ++i) {
          size_t part_size =
              static_cast<size_t>(static_cast<double>(size)/num_servers*(i+1)) -
              static_cast<size_t>(static_cast<double>(size)/num_servers*i);
          ps::Key ps_key = krs[i].begin() + key;
          CHECK_LT(ps_key, krs[i].end());
          pskv.keys.push_back(ps_key);
          pskv.lens.push_back(part_size);
          pskv.size += part_size;
        }
        CHECK_EQ(static_cast<size_t>(pskv.size), size);
      }
    }
    return pskv;
  }

  /**
   * \brief for worker to push and pull data
   */
  ps::KVWorker<char>* ps_worker_ = nullptr;

  /**
   * \brief the server handle
   */
  PSStoreDistServer* server_ = nullptr;

  std::unordered_map<int, size_t> cache_;

  size_t big_bound_ = 1000 * 1000;
};

}  // namespace psstore
}  // namespace tensorflow


#endif  // TENSORFLOW_PS_PSSTORE_DIST_H_

/*!
 * From dmlc/mxnet
 */
#ifndef TENSORFLOW_PS_PSSTORE_DIST_SERVER_H_
#define TENSORFLOW_PS_PSSTORE_DIST_SERVER_H_
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "ps/ps.h"
#include "./psstore.h"

namespace tensorflow {
namespace psstore {
static const int kStopServer = -1;
static const int kSyncMode = -2;
static const int kInitUpdater = -3;

/**
 * \brief executor runs a function using the thread called \ref Start
 */
class Executor {
 public:
  /**
   * \brief start the executor
   */
  void Start() {
	  std::unique_lock<std::mutex> lk(mu_);
	  while (true) {
	    cond_.wait(lk, [this]{return !queue_.empty();});
	    Block blk = std::move(queue_.front());
	    queue_.pop();
	    lk.unlock();
	    if (blk.f) {
	      blk.f(); blk.p->set_value();
	    } else {
	      blk.p->set_value(); break;
	    }
	    lk.lock();
	  }
  }

  /**
   * \brief function
   */
  typedef std::function<void()> Func;

  /**
   * \brief let the thread called \ref Start to exec a function. threadsafe
   */
  void Exec(const Func& func) {
	  Block blk(func);
    auto fut = blk.p->get_future();
	  {
	    std::lock_guard<std::mutex> lk(mu_);
	    queue_.push(std::move(blk));
	    cond_.notify_one();
	  }
	  fut.wait();
  }

  /**
   * \brief stop the thread, threadsafe
   */
  void Stop() {
    Exec(Func());
  }

 private:
  struct Block {
  explicit Block(const Func& func) : f(func), p(std::make_shared<std::promise<void>>()) { }
    Func f;
    std::shared_ptr<std::promise<void>> p;
  };
  std::queue<Block> queue_;
  std::mutex mu_;
  std::condition_variable cond_;
};

class PSStoreDistServer {
 public:
  PSStoreDistServer() {
    using namespace std::placeholders;
	  ps_server_ = new ps::KVServer<char>(0);
	  static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
	      std::bind(&PSStoreDistServer::CommandHandle, this, _1, _2));
	  ps_server_->set_request_handle(
	      std::bind(&PSStoreDistServer::DataHandle, this, _1, _2, _3));
	  sync_mode_ = false;
  }

  ~PSStoreDistServer() {
    delete ps_server_;
  }

  void set_controller(const PSStore::Controller& controller) {
    CHECK(controller);
    controller_ = controller;
  }

  void set_updater(const PSStore::Updater& updater)  {
    CHECK(updater);
    updater_ = updater;
  }

  /**
   * \brief blocked until received the command \a kSyncMode
   */
  void Run() {
    exec_.Start();
  }

 private:
  template<typename T>
  void Sub(Tensor& var, const std::vector<Tensor>& grads) {
    int size = grads.size();
    if (size > 0) {
      //Tensor a(grads[0].dtype(), grads[0].shape());
      Tensor a(grads[0]);
      auto ta = a.flat<T>();
      //ta = grads[0].flat<T>();
      //VLOG(0) << "for test ta begin " << a.DebugString();
      for (int i=1; i<size; i++) {
        ta += grads[i].flat<T>();
      }
      //VLOG(0) << "for test ta after " << a.DebugString();
      var.flat<T>() -= ta / (static_cast<T>(size));
      //VLOG(0) << "for test var " << var.DebugString();
    }
  }

  void CommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    std::string res_body;
	  if (recved.head == kStopServer) {
	    exec_.Stop();
	  } else if (recved.head == kSyncMode) {
	    sync_mode_ = true;
	    VLOG(0) << "Sync mode " << sync_mode_;
	  } else if (recved.head == kInitUpdater) {
	    std::string data_type_name = recved.body;
	    VLOG(0) << "InitUpdater args " << data_type_name;
	    if (typeid(float).name() == data_type_name) {
	      updater_ = std::bind(&PSStoreDistServer::Sub<float>, this, std::placeholders::_1, std::placeholders::_2);
	    } else if (typeid(double).name() == data_type_name) {
	      updater_ = std::bind(&PSStoreDistServer::Sub<double>, this, std::placeholders::_1, std::placeholders::_2);
	    } else {
	      res_body = "set updater function has error: unknown data type name " + data_type_name;
	      VLOG(0) << res_body;
	    }
	  } else {
	    // let the main thread to execute ctrl, which is necessary for python
	    exec_.Exec([this, recved]() {
	      CHECK(controller_);
	      controller_(recved.head, recved.body);
	    });
	  }
	  app->Response(recved, res_body);
  }

  void DataHandle(const ps::KVMeta& req_meta,
                  const ps::KVPairs<char>& req_data,
                  ps::KVServer<char>* server) {
	  // do some check
	  CHECK_EQ(req_data.keys.size(), (size_t)1);
	  if (req_meta.push) {
	    CHECK_EQ(req_data.lens.size(), (size_t)1);
	    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
	  }

	  int key = DecodeKey(req_data.keys[0]);
	  auto& stored = store_[key];
    if (req_meta.push) {
      TensorProto tp;
	    tp.ParseFromArray(req_data.vals.data(), req_data.vals.size());
	    Tensor recved;
	    bool res = recved.FromProto(tp);
	    if (!res) {
	      VLOG(0) << "Parse received data failed!";
	      return;
	    }
	    if (!stored.IsInitialized()) {
	      // model's parameters initialization
	      stored = recved;
	      server->Response(req_meta);
	    } else if (sync_mode_) {
	      // synced push
	      auto& merge_buf = merge_buf_[key];
	      merge_buf.request.push_back(req_meta);
	      merge_buf.buf.push_back(recved);

	      if (merge_buf.request.size() == (size_t)ps::NumWorkers()) {
	        exec_.Exec([this, key, &merge_buf, &stored](){
	            CHECK(updater_);
	            updater_(stored, merge_buf.buf);
	        });
	        for (const auto& req : merge_buf.request) {
	          server->Response(req);
	        }
	        merge_buf.request.clear();
	        merge_buf.buf.clear();
	      }
	    } else {
	      exec_.Exec([this, key, &recved, &stored](){
	          CHECK(updater_);
	          std::vector<Tensor> v;
	          v.push_back(recved);
	          updater_(stored, v);
	      });
	      server->Response(req_meta);
	    }
	  } else {
	    // pull
	    ps::KVPairs<char> response;
	    CHECK(stored.IsInitialized()) << "init " << key << " first";
	    TensorProto tp;
	    stored.AsProtoField(&tp);
	    int size = tp.ByteSize();
	    char *data = new char[size];
	    tp.SerializeToArray(data, size);
	    ps::SArray<char> vals(data, size, true);
	    response.keys = req_data.keys;
	    response.lens = {size};
	    response.vals = vals;
	    //VLOG(0) << "for test pull key " << key << " value " << stored.DebugString();
	    server->Response(req_meta, response);
	  }
  }

  void ApplyGradientDescent(int key, const std::vector<Tensor>& merged, Tensor* stored) {

  }

  inline int DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
  }

  /**
   * \brief user defined
   */
  bool sync_mode_ = false;
  PSStore::Controller controller_;
  PSStore::Updater updater_;

  std::unordered_map<int, Tensor> store_;

  struct MergeBuf {
    std::vector<ps::KVMeta> request;
    std::vector<Tensor> buf;
  };
  std::unordered_map<int, MergeBuf> merge_buf_;

  Executor exec_;

  ps::KVServer<char>* ps_server_;
};


}  // namespace psstore
}  // namespace tensorflow

#endif  // TENSORFLOW_PS_PSSTORE_DIST_SERVER_H_

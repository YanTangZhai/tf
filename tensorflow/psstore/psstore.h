/*!
 * From dmlc/mxnet
 */

#ifndef TENSORFLOW_PS_PSSTORE_H_
#define TENSORFLOW_PS_PSSTORE_H_
#include <vector>
#include <string>
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
namespace psstore {
/*!
 * \brief distributed ps key-value store
 *
 * A distributed ps key-value store for data synchronization over multiple
 * devices/machines. It support user-defined updater.
 */
class PSStore {
 public:
  /*! \brief virtual destructor */
  virtual ~PSStore() {}

  /**
   * \brief the prototype of user-defined updater
   */
  typedef std::function<void(Tensor&, std::vector<Tensor>&)> Updater;

  /*!
   * \brief Factory function to create a new PSStore.
   * \param type The type of the psstore,
   *   - 'local' or 'local_update_cpu' or 'local_allreduce_cpu'
   *       multi-devices on a single machine. can be also
   *   - 'device' or 'local_allreduce_device' : same to local but use gpus for kv
   *       allreduce
   *   - 'dist_*' : multi-machines
   * \return a new created KVStore.
   */

  static PSStore* Create(const char *type = "local", const std::string& args = "");

  /*!
   * \return PSStore singleton.
   */
  static PSStore* Get(const char *type, const std::string& args = "");

  static std::shared_ptr<PSStore> _GetSharedRef(const char *type, const std::string& args = "");

  virtual void Run() = 0;

  /**
   * \brief return the type
   */
  inline const std::string& type() { return type_; }

  /*!
   * \brief Initialize a list of key-value pair to the store.
   *
   * One must initalize the key before \ref Push and \ref Pull, and a key
   * should be only initialized once
   *
   * It returns after data have been initialized successfully.
   *
   * For multiple workers, all workers must call \ref Init. But only worker 0
   * (get_rank() == 0)'s values are used for initialization. So others' values
   * can be empty (but not keys). This function blocks until all workers are
   * finished. That means, any worker can push and pull on the keys now.
   *
   * \param keys a list of unique keys
   * \param values a list of values
   */
  //virtual void Init(const std::vector<int>& keys,
  //                  const std::vector<Tensor>& values) = 0;
  virtual void Init(const int key,
                    const Tensor& value) = 0;

  virtual void InitUpdater(const std::string& args) = 0;

  /*!
   * \brief push a list of key-value pairs into the store
   *
   * If a key appears mulitple times in \a keys, then the according values will
   * be aggregated (summed) before pushing.
   *
   * The (aggregated) values are merged into the store one by one
   *
   * \code
   * updater(key, value, &value_in_store);
   * \endcode
   *
   * One can set a user-defined updater by \ref set_updater. The default updater
   * is Assign.
   *
   * This function returns after adding a push operator to the engine. Any
   * following operator requiring writing value will be blocked until the
   * actual push is finished. One can wait the push is finished by
   *
   * - when type == "local"
   * \code
   * for (auto& v : values) v.WaitToWrite()
   * \endcode
   *
   * - when type == "dist"
   * \code
   * Wait(keys);
   * \endcode
   *
   * One must call Init() on every key before. And the value NDArray should be
   * always has the same shape as being inited.
   *
   * \param keys the list of keys
   * \param values the list of values
   * \param priority Priority of the action.
   */
  //virtual void Push(const std::vector<int>& keys,
  //                  const std::vector<Tensor>& values,
  //                  int priority = 0)  = 0;
  virtual int Push(const int key,
                    const Tensor& value,
                    int priority = 0) = 0;

  /*!
   * \brief pull a list of key-value pairs from the store
   *
   * One must call Init() on \a key before. And \a value should be pre-allocated
   *
   * This function returns after adding a pull operator to the engine. Any
   * following operator requiring reading value will be blocked until the
   * actual pull is finished. One can wait the pull is finished by
   *
   * - when type == "local"
   * \code
   * for (auto& v : values) v.WaitToRead()
   * \endcode
   *
   * - when type == "dist"
   * \code
   * Wait(keys);
   * \endcode
   *
   * \param keys the list of keys
   * \param values the list of buffers for the pulled data, they should be preallocated
   * \param priority Priority of the action.
   */
  //virtual void Pull(const std::vector<int>& keys,
  //                  const std::vector<Tensor*>& values,
  //                  int priority = 0) = 0;
  virtual bool Pull(const int key,
                    Tensor& value,
                    int priority = 0) = 0;

  /*!
   * \brief set an updater
   *
   * Given a key, assume \a x is the received (pushed) value and \a y is the
   * value stored on the store node. The store updates \a y by `h(x, &y)`. The
   * default \a h is ASSIGN, namely `*y = x`.
   *
   * \param updater user-defined updater, default is assign
   */
  virtual void set_updater(const Updater& updater) {
    updater_ = updater;
  }

  /**
   * \brief the prototype of a server controller
   */
  typedef std::function<void(int, const std::string&)> Controller;

  virtual void set_controller(const Controller& controller) {
    CHECK(controller) << "invalid controller";
    controller_ = controller;
  }

  /******************************************************
   * the following are used for multi-machines.
   ******************************************************/

  /**
   * \return whether or not this process is a worker node.
   *
   * Always returns true when type == "local"
   */
  static bool IsWorkerNode() {
    char* role_str = getenv("TENSORFLOW_ROLE");
    return (role_str == nullptr) || (!strcmp(role_str, "worker"));
  }

  /**
   * \return whether or not this process is a server node.
   *
   * Always returns false when type == "local"
   */
  static bool IsServerNode() {
    char* role_str = getenv("TENSORFLOW_ROLE");
    return (role_str != nullptr) && (!strcmp(role_str, "server"));
  }


  /**
   * \return whether or not this process is a scheduler node.
   *
   * Always returns false when type == "local"
   */
  static bool IsSchedulerNode() {
    char* role_str = getenv("TENSORFLOW_ROLE");
    return (role_str != nullptr) && (!strcmp(role_str, "scheduler"));
  }

  /*!
   * \return The rank of this node in its group, which is in [0,
   * GroupSize).
   *
   * Always return 0 when type == "local"
   */
  virtual int get_rank() const {
    return 0;
  }

  /*!
   * \return The number of worker nodes
   */
  virtual int get_group_size() const {
    return 1;
  }

  /*!
   * \brief global barrier among all worker machines
   *
   * But note that, this functions only blocks the main thread of workers until
   * all of them are reached this point. It doesn't guarantee that all
   * operations issued before are actually finished, such as \ref Push and \ref Pull.
   */
  virtual void Barrier() { }

  /**
   * \brief Send a command to all server nodes
   *
   * Send a command to all server nodes, which will make each server node run
   * \a controller
   *
   * This function returns after the command has been executed in all server nodes
   *
   * \param cmd_id the head of the command
   * \param cmd_body the body of the command
   */
  virtual void SendCommandToServers(int cmd_id, const std::string& cmd_body) { }

 protected:
  /**
   * \brief the user-defined  updater
   */
  Updater updater_;
  Controller controller_;

  /**
   * \brief the kvstore type
   */
  std::string type_;
};

}  // namespace psstore
}  // namespace tensorflow
#endif  // TENSORFLOW_PS_PSSTORE_H_

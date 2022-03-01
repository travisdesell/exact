#include <memory>
using std::unique_ptr;

#include <utility>
using std::move;

#include <vector>
using std::vector;

#include "msg.hxx"

#include "../common/archipelago_config.hxx"


class ArchipelagoNode {
 protected:
  ArchipelagoConfig &config;
  vector<node_index_type> parents, children, neighbors;
  int terminate_count = 0;

  void send_msg_to(unique_ptr<Msg> msg, node_index_type dst);
  void send_msg_to_children(unique_ptr<Msg> msg);
  void send_msg_to_neighbors(unique_ptr<Msg> msg);
  // Be sure ONLY propagate w/ one parent
  void send_msg_to_parents(unique_ptr<Msg> msg);
  unique_ptr<Msg> recieve_msg()

 public:
  ArchipelagoNode(ArchipelagoConfig &config);
  virtual ~ArchipelagoNode();

  void run() {
    while (terminate_count != parents.size() || parents.size() == 0) {
      unique_ptr<Msg> msg = ArchipelagoNode::recieve_msg();

      if (TerminateMsg *tmsg = dynamic_cast<TerminateMsg *>(msg.get()); tmsg != nullptr)
        terminate_count += 1;
      else
        process_msg(move(msg));
    }
  }
  virtual void process_msg(unique_ptr<Msg> msg) = 0;
};

class ArchipelagoIsland : public ArchipelagoNode {
};

class ArchipelagoManager : public ArchipelagoNode {
};

class ArchipelagoMaster : public ArchipelagoNode {
};

class ArchipelagoWorker : public ArchipelagoNode {
};

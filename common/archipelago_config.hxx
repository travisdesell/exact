#ifndef ARCHIPELAGO_CONFIG_HXX
#define ARCHIPELAGO_CONFIG_HXX

#include <stddef.h>

#include <memory>
using std::unique_ptr;

#include <optional>
using std::optional;

#include <vector>
using std::vector;

#include <variant>
using std::variant;

#include <string>
using std::string;

#include <map>
using std::map;

typedef ssize_t node_id;
class ArchipelagoConfig {
 public:
   const node_id master_id;

   // Maps node id to a list of node id that it should be connected to.
   const vector<vector<node_id>> connections;

   enum node_type { WORKER, ISLAND, MANAGER, MASTER };

   const vector<node_type> node_types;

   ArchipelagoConfig(node_id master_id, vector<vector<node_id>> connections, vector<node_type> node_types);

   // Check to see if a config is good:
   // - All workers have a manager.
   // - All nodes can reach the master
   void eval_config();
};

struct Token {
  enum token_type {
    COLON,        // :
    ID,           // ([a-z]|[A-Z])+([0-9]|_|[a-z][A-Z])*
    INT,          // [0-9]+
    CONNECTION,   // ->
    DASH,         // -
    COMMA,        // ,
    KW_MASTER,    // master
    KW_MANAGERS,  // managers
    KW_ISLANDS,   // islands
    KW_WORKERS,   // workers
    KW_TOPOLOGY,  // topology
    KW_N,         // n
    MUL,          // *
    DIV,          // /
    ADD,          // +
    SUB,          // -
    MOD,          // %
    EQ,           // =
    KW_PARTITION, // partition 1 of 100 by 25
    KW_OF,        // of
    KW_BY,        // by
    LPAREN,       // (
    RPAREN,       // )
  };


  token_type ty;
  string data;

  Token(token_type ty, string data, int32_t line, int32_t column);
};

// Take characters from string starting from index while a condition is true (or EOF is reached).
// Return the resulting substring.
string take_while(string file, int &index);
void skip_whitespace(string file, int &index);

optional<Token> next_token(string file, int &index);

/** GRAMMAR
 *
 * sum_expr = product_expr ((ADD | SUB) product_expr)*
 * product_expr = partition_expr ((MUL | DIV) partition_expr)*
 * partition_expr = PARTITION_KW expr_inner OF_KW expr_inner BY_KW expr_inner | expr_inner
 * expr_inner = ID | INT | KW_MASTER | LPAREN sum_expr RPAREN
 * expr = sum_expr
 *
 * node_ref = expr DASH expr | expr
 * node_ref_list = node_ref (COMMA node_ref)+
 * abstract_node_ref = 
 *    KW_MASTER 
 *  | KW_MANAGERS
 *  | KW_ISLANDS
 *  | KW_WORKER
 *  | node_ref
 *
 * abstract_node_ref_list = abstract_node_ref (COMMA abstract_node_ref)*
 *
 * connection = abstract_node_ref_list CONNECTION abstract_node_ref_list
 * 
 * role_assignment = 
 *    KW_MASTER COLON EXPR
 *  | (KW_MANAGERS | KW_ISLANDS | KW_WORKERS) COLON abstract_node_ref_list
 *
 * topology = KW_TOPOLOGY ID abstract_node_ref_list
 *
 * var_assignment = ID EQ EXPR
 * 
 **/

enum special_group { MASTER, MANAGERS, ISLANDS, WORKERS } group;
class Env {
  vector<vector<node_id>> connections;
  map<string, node_id> vars;
};

class AST {
 public:
  AST(int32_t line, int32_t column);
  virtual ~AST();
};

class Expr : public AST {
 public:
  Expr(int32_t line, int32_t column);
  virtual ~Expr() = 0;

  virtual ssize_t eval(Env &env) = 0;
};

class ArithExpr : public Expr {
 unique_ptr<Expr> l, r;
 enum arith_op { MUL, DIV, MOD, ADD, SUB } op;

 public:
  ArithExpr(unique_ptr<Expr> l, unique_ptr<Expr> r, arith_op op, int32_t line, int32_t column);
  virtual ~ArithExpr();
  
  virtual ssize_t eval(Env &env);
};

class IdExpr : public Expr {
  string id;
 public:
  IdExpr(string id, int32_t line, int32_t column);
  virtual ~IdExpr();

  virtual ssize_t eval(Env &env);
};

class PartitionExpr : public Expr {
  ssize_t index, by, of;

 public:
  PartitionExpr(ssize_t index, ssize_t by, ssize_t of, int32_t line, int32_t column);
  virtual ~PartitionExpr();

  virtual ssize_t eval(Env &env);
};

class ConstExpr : public Expr {
  ssize_t value;

 public:
  ConstExpr(ssize_t value, int32_t line, int32_t column);
  virtual ~ConstExpr();

  virtual ssize_t eval(Env &env);
};

class NodeRef : public AST {
 public:
  NodeRef(int32_t line, int32_t column);
  virtual ~NodeRef() = 0;

  virtual void get(vector<node_id> &indices) = 0;
};

class NodeRange : public NodeRef {
  unique_ptr<Expr> start, end;

 public:
  NodeRange(unique_ptr<Expr> start, unique_ptr<Expr> end, int32_t line, int32_t column);
  virtual ~NodeRange();

  virtual void get(vector<node_id> &indices);
};

class SingletonNode : public NodeRef {
  unique_ptr<Expr> node;

 public:
  SingletonNode(unique_ptr<Expr> node, int32_t line, int32_t column);
  virtual ~SingletonNode();

  virtual void get(vector<node_id> &indices);
};

class SpecialNodeRef : public NodeRef {
  special_group group; 
 public:
  SpecialNodeRef(special_group group, int32_t line, int32_t column);
  virtual ~SpecialNodeRef();

  virtual void get(vector<node_id> &indices);
};

class Statement : public Expr {
  protected:
   static vector<node_id> eval_refs(vector<NodeRef> &refs);

  public:
   Statement(int32_t line, int32_t column);
   virtual ~Statement() = 0;

   virtual void execute(Env &env) = 0;
};

class ConnectionStatement : public Statement {
  vector<NodeRef> from, to;

 public:
  ConnectionStatement(vector<NodeRef> from, vector<NodeRef> to, int32_t line, int32_t column);
  virtual ~ConnectionStatement();

  virtual void execute(Env &env);
};

class RoleStatement : public Statement {
  vector<NodeRef> members;
  special_group group;

 public:
  RoleStatement(vector<NodeRef> members, special_group group, int32_t line, int32_t column);
  virtual ~RoleStatement();

  virtual void execute(Env &env);
};

class TopologyStatement : public Statement {
  enum topology_type { RING } topology;

 public:
  TopologyStatement(topology_type topology, vector<NodeRef> members, int32_t line, int32_t column);
  virtual ~TopologyStatement();

  virtual void execute(Env &env);
};

class AssignmentStatement : public Statement {
  string id;
  unique_ptr<Expr> value;
 
 public:
  AssignmentStatement(string id, unique_ptr<Expr> value, int32_t line, int32_t column);
  virtual ~AssignmentStatement();

  virtual void execute(Env &env);
};

class ArchipelagoParser {

};

#endif // ARCHIPELAGO_CONFIG_HXX

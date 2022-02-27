#ifndef ARCHIPELAGO_CONFIG_HXX
#define ARCHIPELAGO_CONFIG_HXX

#include <stddef.h>

#include <functional>
using std::function;

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

#include <unordered_map>
using std::unordered_map;

#include <map>
using std::map;

#include "log.hxx"

typedef int32_t node_index_type;

struct Token {
  enum token_type {
    COLON,        // : X
    ID,           // ([a-z]|[A-Z])+([0-9]|_|[a-z][A-Z])* X
    INT,          // [0-9]+ X
    CONNECTION,   // -> X
    COMMA,        // , X
    KW_MASTER,    // master X
    KW_MANAGERS,  // managers X
    KW_ISLANDS,   // islands X
    KW_WORKERS,   // workers X
    KW_TOPOLOGY,  // topology X
    KW_N_NODES,   // n_nodes X
    MUL,          // * X
    DIV,          // / X
    ADD,          // + X
    SUB,          // - X
    MOD,          // % X
    EQ,           // = X
    KW_PARTITION, // partition 1 of 100 by 25 X
    KW_OF,        // of X
    KW_BY,        // by X
    KW_THROUGH,   // 1 through 4  inclusive range X
    KW_UNTIL,     // 1 until 4    exclusive range X
    LPAREN,       // ( X
    RPAREN,       // ) X
    COMMENT,      // # X
  };

  static const unordered_map<token_type, string> display_map;


  token_type ty;
  string data;
  int32_t line, column;

  Token(token_type ty, string data, int32_t line, int32_t column);
 
 public:
  string to_string() const;
  string debug() const;
};

// Take characters from string starting from index while a condition is true (or EOF is reached).
// Return the resulting substring.

class Tokenizer {
  static const unordered_map<string, Token::token_type> kw_map;

  const string text;
  size_t index;
  int32_t line, column;

  string& take_while(function<bool (char)>& cond, string &val);
  string& take_until(function<bool (char)>& cond, string &val);
  void skip_whitespace();
  optional<char> peek();
  optional<char> pop();

  Token make_token(Token::token_type ty, string data);
  Token ident();
  Token number();
  Token error(string description);
  Token comment();

 public:
  Tokenizer(string text);
  
  vector<Token> tokenize();
  optional<Token> next_token();
};

/** GRAMMAR
 *
 * sum_expr = product_expr ((ADD | SUB) product_expr)*
 * product_expr = partition_expr ((MUL | DIV) partition_expr)*
 * partition_expr = PARTITION_KW expr_inner OF_KW node_range BY_KW expr_inner | expr_inner
 * expr_inner = ID | KW_N_NODES | INT | KW_MASTER | LPAREN sum_expr RPAREN
 * expr = sum_expr
 *
 * node_range = expr (KW_UNTIL | KW_THROUGH) expr 
 * node_ref = node_range | expr
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
 * topology = KW_TOPOLOGY ID COLON abstract_node_ref_list
 *
 * var_assignment = ID EQ EXPR
 * 
 **/

enum special_group { MASTER, MANAGERS, ISLANDS, WORKERS };
unordered_map<Token::token_type, special_group> special_group_map = {
  {Token::KW_MASTER, MASTER},
  {Token::KW_ISLANDS, ISLANDS},
  {Token::KW_MANAGERS, MANAGERS},
  {Token::KW_WORKERS, WORKERS}
};

struct Env {
  vector<vector<node_index_type>> connections;
  map<string, node_index_type> vars;

  node_index_type error(string message);
};

class AST {
 public:
  const int32_t line, column;
  AST(int32_t line, int32_t column);
  virtual ~AST();
};

class Expr : public AST {
 public:
  Expr(int32_t line, int32_t column);
  virtual ~Expr();

  virtual node_index_type eval(Env &env) = 0;
};

class ArithExpr : public Expr {
  unique_ptr<Expr> l, r;
 public:
  const enum arith_op { MUL, DIV, MOD, ADD, SUB } op;
  
  ArithExpr(unique_ptr<Expr> l, unique_ptr<Expr> r, arith_op op, int32_t line, int32_t column);
  virtual ~ArithExpr() = default;

  virtual node_index_type eval(Env &env);
};

class IdExpr : public Expr {
  string id;
 public:
  IdExpr(string id, int32_t line, int32_t column);
  virtual ~IdExpr() = default;

  virtual node_index_type eval(Env &env);
};

class PartitionExpr : public Expr {
  unique_ptr<Expr> index, lower, upper, divisor;
  bool inclusive;

 public:
  PartitionExpr(unique_ptr<Expr> index, unique_ptr<Expr> lower, unique_ptr<Expr> upper, bool inclusive,
                unique_ptr<Expr> divisor, int32_t line, int32_t column);
  virtual ~PartitionExpr() = default;

  virtual node_index_type eval(Env &env);
};

class KWExpr : public Expr {
 public:
  const enum KWExprTy { MASTER, N_NODES } ty;
  KWExpr(KWExprTy ty, int32_t line, int32_t column);
  virtual ~KWExpr() = default;

  virtual node_index_type eval(Env &env);
};

class ConstExpr : public Expr {
  node_index_type value;

 public:
  ConstExpr(node_index_type value, int32_t line, int32_t column);
  virtual ~ConstExpr() = default;

  virtual node_index_type eval(Env &env);
};

class NodeRef : public AST {
 public:
  NodeRef(int32_t line, int32_t column);
  virtual ~NodeRef() = default;

  virtual void get(vector<node_index_type> &indices) = 0;
};

class NodeRange : public NodeRef {
  unique_ptr<Expr> start, end;
  bool inclusive;

 public:
  NodeRange(unique_ptr<Expr> start, unique_ptr<Expr> end, bool inclusive, int32_t line, int32_t column);
  virtual ~NodeRange() = default;

  virtual void get(vector<node_index_type> &indices);
};

class SingletonNode : public NodeRef {
  unique_ptr<Expr> node;

 public:
  SingletonNode(unique_ptr<Expr> node, int32_t line, int32_t column);
  virtual ~SingletonNode() = default;

  virtual void get(vector<node_index_type> &indices);
};

class SpecialNodeRef : public NodeRef {
  special_group group;

 public:
  SpecialNodeRef(special_group group, int32_t line, int32_t column);
  virtual ~SpecialNodeRef() = default;

  virtual void get(vector<node_index_type> &indices);
};

class Statement : public AST {
 protected:
  static vector<node_index_type> eval_refs(vector<NodeRef> &refs);

 public:
  Statement(int32_t line, int32_t column);
  virtual ~Statement();

  virtual void execute(Env &env) = 0;
};

class ConnectionStatement : public Statement {
  vector<unique_ptr<NodeRef>> from, to;

 public:
  ConnectionStatement(vector<unique_ptr<NodeRef>> from, vector<unique_ptr<NodeRef>> to, int32_t line, int32_t column);
  virtual ~ConnectionStatement() = 0;

  virtual void execute(Env &env);
};

class RoleStatement : public Statement {
  vector<unique_ptr<NodeRef>> members;
  special_group group;

 public:
  RoleStatement(vector<unique_ptr<NodeRef>> members, special_group group, int32_t line, int32_t column);
  RoleStatement(unique_ptr<Expr> member, int32_t line, int32_t column); // For master only
  virtual ~RoleStatement() = 0;

  virtual void execute(Env &env);
};

class TopologyStatement : public Statement {
  enum topology_type { RING } topology;
  vector<unique_ptr<NodeRef>> members;

 public:
  TopologyStatement(topology_type topology, vector<unique_ptr<NodeRef>> members, int32_t line, int32_t column);
  virtual ~TopologyStatement() = 0;
  
  static topology_type get_topology_type(string str);

  virtual void execute(Env &env);
};

class AssignmentStatement : public Statement {
  string id;
  unique_ptr<Expr> value;

 public:
  AssignmentStatement(string id, unique_ptr<Expr> value, int32_t line, int32_t column);
  virtual ~AssignmentStatement() = 0;

  virtual void execute(Env &env);
};

class Parser {
  vector<Token> tokens;
  int32_t index;

  const Token *peek(int lookahead = 0);
  template <unsigned int N>
  const std::array<const Token *, N> peek();
  
  optional<Token> pop(int lookahead = 0);
  template <unsigned int N>
  std::array<optional<Token>, N> pop();
 
  template <unsigned int N>
  void expect(std::array<const Token *, N> &toks, std::array<Token::token_type, N> &expected);

  void error(string msg);

 public:
  Parser(vector<Token> tokens);

  vector<unique_ptr<Statement>> parse();

  unique_ptr<Statement> parse_statement();
  unique_ptr<Statement> parse_topology();
  unique_ptr<Statement> parse_var_assignment();
  unique_ptr<Statement> parse_role_assignment();
  unique_ptr<Statement> parse_connection();
  
  vector<unique_ptr<NodeRef>> parse_abstract_node_ref_list();
  vector<unique_ptr<NodeRef>> parse_node_ref_list();
  
  unique_ptr<NodeRef> parse_abstract_node_ref();
  unique_ptr<NodeRef> parse_node_ref();
  
  unique_ptr<Expr> parse_expr();
  unique_ptr<Expr> parse_expr_inner();
  unique_ptr<Expr> parse_partition_expr();
  unique_ptr<Expr> parse_product_expr();
  unique_ptr<Expr> parse_sum_expr();
};

class ArchipelagoConfig {
 public:
  const node_index_type master_id;

  // Maps node id to a list of node id that it should be connected to.
  const vector<vector<node_index_type>> connections;

  enum node_type { WORKER, ISLAND, MANAGER, MASTER };

  const vector<node_type> node_types;

  ArchipelagoConfig(node_index_type master_id, vector<vector<node_index_type>> connections, vector<node_type> node_types);

  // Check to see if a config is good:
  // - All islands have an manager
  // - All managers are connected to the master
  // - All nodes can reach the master
  // - All islands have workers
  void eval_config();
};

#endif // ARCHIPELAGO_CONFIG_HXX

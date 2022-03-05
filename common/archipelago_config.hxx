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
    COLON,         // : X
    ID,            // ([a-z]|[A-Z])+([0-9]|_|[a-z][A-Z])* X
    INT,           // [0-9]+ X
    CONNECTION,    // -> X
    COMMA,         // , X
    KW_MASTER,     // master X
    KW_MANAGERS,   // managers X
    KW_ISLANDS,    // islands X
    KW_WORKERS,    // workers X
    KW_TOPOLOGY,   // topology X
    KW_N_NODES,    // n_nodes X
    MUL,           // * X
    DIV,           // / X
    ADD,           // + X
    SUB,           // - X
    MOD,           // % X
    EQ,            // = X
    KW_PARTITION,  // partition 1 of 100 by 25 X
    KW_OF,         // of X
    KW_BY,         // by X
    KW_THROUGH,    // 1 through 4  inclusive range X
    KW_UNTIL,      // 1 until 4    exclusive range X
    LPAREN,        // ( X
    RPAREN,        // ) X
    COMMENT,       // # X
    CB_OPEN,       // {
    CB_CLOSE,      // }
    KW_IN,         // in
    KW_FOR,        // for
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

  string &take_while(function<bool(char)> &cond, string &val);
  string &take_until(function<bool(char)> &cond, string &val);
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

enum node_role { MASTER = 0, MANAGERS = 1, ISLANDS = 2, WORKERS = 3 };

struct Env {
  static const unordered_map<Token::token_type, node_role> node_role_map;
  static const unordered_map<node_role, string> node_role_string_map;

  // should be n_nodes by n_nodes matrix.
  // nodes F and T are connected if connections[F][T] is true.
  map<string, node_index_type> vars;

  vector<node_role> node_roles;
  vector<vector<bool>> connections;

  node_index_type master;
  const node_index_type n_nodes;

  Env(node_index_type n_node);

  node_index_type error(string message);
  void connect(node_index_type from, node_index_type to);
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
  virtual string to_string() = 0;
};

class ArithExpr : public Expr {
  unique_ptr<Expr> l, r;

 public:
  const enum arith_op { MUL, DIV, MOD, ADD, SUB } op;

  ArithExpr(unique_ptr<Expr> l, unique_ptr<Expr> r, arith_op op, int32_t line, int32_t column);
  virtual ~ArithExpr();

  virtual node_index_type eval(Env &env);
  virtual string to_string();
};

class IdExpr : public Expr {
  string id;

 public:
  IdExpr(string id, int32_t line, int32_t column);
  virtual ~IdExpr();

  virtual node_index_type eval(Env &env);
  virtual string to_string();
};

class PartitionExpr : public Expr {
  unique_ptr<Expr> index, lower, upper, divisor;
  bool inclusive;

 public:
  PartitionExpr(unique_ptr<Expr> index, unique_ptr<Expr> lower, unique_ptr<Expr> upper, bool inclusive,
                unique_ptr<Expr> divisor, int32_t line, int32_t column);
  virtual ~PartitionExpr();

  virtual node_index_type eval(Env &env);
  virtual string to_string();
};

class KWExpr : public Expr {
 public:
  const enum KWExprTy { MASTER, N_NODES } ty;
  KWExpr(KWExprTy ty, int32_t line, int32_t column);
  virtual ~KWExpr();

  virtual node_index_type eval(Env &env);
  virtual string to_string();
};

class ConstExpr : public Expr {
  node_index_type value;

 public:
  ConstExpr(node_index_type value, int32_t line, int32_t column);
  virtual ~ConstExpr();

  virtual node_index_type eval(Env &env);
  virtual string to_string();
};

class NodeRef : public AST {
 public:
  NodeRef(int32_t line, int32_t column);
  virtual ~NodeRef();

  virtual void get(vector<node_index_type> &indices, Env &env) = 0;
  virtual string to_string() = 0;
};

class NodeRange : public NodeRef {
  unique_ptr<Expr> start, end;
  bool inclusive;

 public:
  NodeRange(unique_ptr<Expr> start, unique_ptr<Expr> end, bool inclusive, int32_t line, int32_t column);
  virtual ~NodeRange();

  virtual void get(vector<node_index_type> &indices, Env &env);
  virtual string to_string();
};

class SingletonNode : public NodeRef {
  unique_ptr<Expr> node;

 public:
  SingletonNode(unique_ptr<Expr> node, int32_t line, int32_t column);
  virtual ~SingletonNode();

  virtual void get(vector<node_index_type> &indices, Env &env);
  virtual string to_string();
};

class SpecialNodeRef : public NodeRef {
  node_role group;

 public:
  SpecialNodeRef(node_role group, int32_t line, int32_t column);
  virtual ~SpecialNodeRef();

  virtual void get(vector<node_index_type> &indices, Env &env);
  virtual string to_string();
};

class Statement : public AST {
 protected:
 public:
  Statement(int32_t line, int32_t column);
  virtual ~Statement();

  virtual void execute(Env &env) = 0;
  virtual string to_string() = 0;
};

class ForStatement : public Statement {
  string id;
  vector<unique_ptr<NodeRef>> refs;
  unique_ptr<Statement> statement;

 public:
  ForStatement(string id, vector<unique_ptr<NodeRef>> refs, unique_ptr<Statement> statement, int32_t line,
               int32_t column);
  virtual ~ForStatement();

  virtual void execute(Env &env);
  virtual string to_string();
};

class CompoundStatement : public Statement {
  vector<unique_ptr<Statement>> statements;

 public:
  CompoundStatement(vector<unique_ptr<Statement>> statements, int32_t line, int32_t column);
  virtual ~CompoundStatement();
  virtual void execute(Env &env);
  virtual string to_string();
};

class ConnectionStatement : public Statement {
  vector<unique_ptr<NodeRef>> from, to;

 public:
  ConnectionStatement(vector<unique_ptr<NodeRef>> from, vector<unique_ptr<NodeRef>> to, int32_t line, int32_t column);
  virtual ~ConnectionStatement();

  virtual void execute(Env &env);
  virtual string to_string();
};

class RoleStatement : public Statement {
  vector<unique_ptr<NodeRef>> members;
  node_role group;

 public:
  RoleStatement(vector<unique_ptr<NodeRef>> members, node_role group, int32_t line, int32_t column);
  RoleStatement(unique_ptr<Expr> member, int32_t line, int32_t column);  // For master only
  virtual ~RoleStatement();

  virtual void execute(Env &env);
  virtual string to_string();
};

class TopologyStatement : public Statement {
  enum topology_type { RING } topology;
  vector<unique_ptr<NodeRef>> members;

 public:
  TopologyStatement(topology_type topology, vector<unique_ptr<NodeRef>> members, int32_t line, int32_t column);
  virtual ~TopologyStatement();

  static topology_type get_topology_type(string str);

  virtual void execute(Env &env);
  virtual string to_string();
};

class AssignmentStatement : public Statement {
  string id;
  unique_ptr<Expr> value;

 public:
  AssignmentStatement(string id, unique_ptr<Expr> value, int32_t line, int32_t column);
  virtual ~AssignmentStatement();

  virtual void execute(Env &env);
  virtual string to_string();
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

  void error(string msg, int line);

 public:
  Parser(vector<Token> tokens);

  vector<unique_ptr<Statement>> parse();

  unique_ptr<Statement> parse_statement();
  unique_ptr<Statement> parse_topology();
  unique_ptr<Statement> parse_var_assignment();
  unique_ptr<Statement> parse_role_assignment();
  unique_ptr<Statement> parse_connection();
  unique_ptr<Statement> parse_compound_statement();
  unique_ptr<Statement> parse_for();

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
  const vector<vector<bool>> connections;

  const vector<node_role> node_roles;

  static ArchipelagoConfig from_string(string cfg, int32_t n_nodes,
                                       map<string, node_index_type> &define_map = empty_map);

 private:
  ArchipelagoConfig(node_index_type master_id, vector<vector<bool>> connections, vector<node_role> node_roles);

  static inline map<string, node_index_type> empty_map = {};
  // Check to see if a config is good:
  // - All islands have an manager
  // - All managers are connected to the master
  // - All nodes can reach the master
  // - All islands have workers
  void eval_config();
};

#endif  // ARCHIPELAGO_CONFIG_HXX

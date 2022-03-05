#include "archipelago_config.hxx"

#include <utility>
using std::move;

#include <cstring>

#include <iostream>
#include <sstream>
#include <set>
using std::set;
#include <math.h>

///
/// Token
///

const unordered_map<Token::token_type, string> Token::display_map = {
    {COLON,        "COLON"       },
    {ID,           "ID"          },
    {INT,          "INT"         },
    {CONNECTION,   "CONNECTION"  },
    {COMMA,        "COMMA"       },
    {KW_MASTER,    "KW_MASTER"   },
    {KW_MANAGERS,  "KW_MANAGERS" },
    {KW_ISLANDS,   "KW_ISLANDS"  },
    {KW_WORKERS,   "KW_WORKERS"  },
    {KW_TOPOLOGY,  "KW_TOPOLOGY" },
    {KW_N_NODES,   "KW_N_NODES"  },
    {MUL,          "MUL"         },
    {DIV,          "DIV"         },
    {ADD,          "ADD"         },
    {SUB,          "SUB"         },
    {MOD,          "MOD"         },
    {EQ,           "EQ"          },
    {KW_PARTITION, "KW_PARTITION"},
    {KW_OF,        "KW_OF"       },
    {KW_BY,        "KW_BY"       },
    {KW_THROUGH,   "KW_THROUGH"  },
    {KW_UNTIL,     "KW_UNTIL"    },
    {LPAREN,       "LPAREN"      },
    {RPAREN,       "RPAREN"      },
    {COMMENT,      "COMMENT"     },
    {KW_IN,        "KW_IN"       },
    {KW_FOR,       "KW_FOR"      },
    {CB_OPEN,      "CB_OPEN"     },
    {CB_CLOSE,     "CB_CLOSE"    },
};

Token::Token(token_type ty, string data, int32_t line, int32_t column)
    : ty(ty), data(move(data)), line(line), column(column) {}

string Token::to_string() const {
  std::stringbuf buf;
  std::ostream os(&buf);

  const string &ty_str = display_map.at(ty);
  os << "\"" << data << "\" on line " << line << ":" << column;

  return buf.str();
}

string Token::debug() const {
  std::stringbuf buf;
  std::ostream os(&buf);

  const string &ty_str = display_map.at(ty);
  os << "< " << line << ":" << column << ":" << ty_str << ":\"" << data << "\" >";

  return buf.str();
}

///
/// Tokenizer
///

Tokenizer::Tokenizer(string text) : text(move(text)), index(0), line(1), column(1) {}

optional<char> Tokenizer::pop() {
  if (index < text.size()) {
    column += 1;
    return text[index++];
  } else {
    return std::nullopt;
  }
}

optional<char> Tokenizer::peek() {
  if (index < text.size())
    return text[index];
  else
    return std::nullopt;
}

void Tokenizer::skip_whitespace() {
  while (1) {
    auto x = peek();

    if (x == std::nullopt) return;
    if (!isspace(*x)) return;

    if (*x == '\n') {
      line += 1;
      column = 1;
    }

    index += 1;
  }
}

string &Tokenizer::take_while(function<bool(char)> &condition, string &val) {
  while (1) {
    auto x = peek();

    if (x == std::nullopt) return val;
    if (!condition(*x)) return val;

    val.push_back(*pop());
  }
}

string &Tokenizer::take_until(function<bool(char)> &condition, string &val) {
  while (1) {
    auto x = peek();

    if (x == std::nullopt) return val;
    if (condition(*x)) return val;

    val.push_back(*pop());
  }
}

Token Tokenizer::make_token(Token::token_type ty, string data) { return Token(ty, move(data), line, column); }

const unordered_map<string, Token::token_type> Tokenizer::kw_map = {
    {"master",    Token::KW_MASTER   },
    {"managers",  Token::KW_MANAGERS },
    {"islands",   Token::KW_ISLANDS  },
    {"workers",   Token::KW_WORKERS  },
    {"topology",  Token::KW_TOPOLOGY },
    {"n_nodes",   Token::KW_N_NODES  },
    {"partition", Token::KW_PARTITION},
    {"of",        Token::KW_OF       },
    {"by",        Token::KW_BY       },
    {"through",   Token::KW_THROUGH  },
    {"until",     Token::KW_UNTIL    },
    {"in",        Token::KW_IN       },
    {"for",       Token::KW_FOR      },
};

static function<bool(char)> isidchar_lambda = [](char c) -> bool { return isalnum(c) || c == '_'; };
Token Tokenizer::ident() {
  auto c = peek();
  if (c == std::nullopt || (!isalpha(*c) && *c != '_'))
    return error("Invalid identifier. This is likely an issue with the tokenizer");

  string id;
  take_while(isidchar_lambda, id);

  unordered_map<string, Token::token_type>::const_iterator it;
  if ((it = kw_map.find(id)) != kw_map.end())
    return make_token(it->second, move(id));
  else
    return make_token(Token::ID, move(id));
}

static function<bool(char)> isnum_lambda = [](char c) -> bool { return isdigit(c); };
Token Tokenizer::number() {
  auto c = peek();
  if (c == std::nullopt || !isdigit(*c)) return error("Invalid identifier. This is likely an issue with the tokenizer");

  string n;
  take_while(isnum_lambda, n);

  c = peek();
  if (c != std::nullopt && isalpha(*c))
    return error("Invalid number that ends in a character. Use a space to separate numbers and identifiers!");

  return make_token(Token::INT, move(n));
}

static function<bool(char)> is_newline = [](char c) -> bool { return c == '\n'; };
Token Tokenizer::comment() {
  auto c = peek();
  if (c == std::nullopt || *c != '#')
    return error("Comment should begin with a #. This is likely an issue with the tokenizer");

  string text;
  take_until(is_newline, text);

  return make_token(Token::COMMENT, text);
}

Token Tokenizer::error(string reason) {
  Log::info("Encountered error while parsing archipelago config: %s", reason.c_str());
  exit(1);
}

optional<Token> Tokenizer::next_token() {
  skip_whitespace();
  auto c = peek();

  if (c == std::nullopt) return std::nullopt;

  switch (*c) {
    case '{':
      pop();
      return make_token(Token::CB_OPEN, "{");
    case '}':
      pop();
      return make_token(Token::CB_CLOSE, "}");
    case ':':
      pop();
      return make_token(Token::COLON, ":");
    case '+':
      pop();
      return make_token(Token::ADD, "+");
    case '*':
      pop();
      return make_token(Token::MUL, "*");
    case '/':
      pop();
      return make_token(Token::DIV, "/");
    case '%':
      pop();
      return make_token(Token::MOD, "%");
    case '=':
      pop();
      return make_token(Token::EQ, "=");
    case '(':
      pop();
      return make_token(Token::LPAREN, "(");
    case ')':
      pop();
      return make_token(Token::RPAREN, ")");
    case ',':
      pop();
      return make_token(Token::COMMA, ",");
    case '#':
      return comment();
    case '-': {
      pop();
      auto c2 = peek();
      if (c2 == std::nullopt || *c2 != '>') {
        return make_token(Token::SUB, "-");
      } else {
        pop();
        return make_token(Token::CONNECTION, "->");
      }
    }
    default:
      if (isalpha(*c) || *c == '_')
        return ident();
      else if (isdigit(*c))
        return number();
      else if (*c == EOF)
        return std::nullopt;
      else
        return error("Unexpected character: " + std::to_string(*c));
  }
}

vector<Token> Tokenizer::tokenize() {
  vector<Token> tokens;

  optional<Token> tok;

  while ((tok = next_token()) != std::nullopt) {
    if (tok->ty == Token::COMMENT) continue;
    tokens.push_back(*tok);
  }

  return tokens;
}

///
/// Parser
///

Parser::Parser(vector<Token> tokens) : tokens(move(tokens)), index(0) {}

void Parser::error(string msg, int line) {
  Log::error(("Encountered error on line %d: " + msg + "\n").c_str(), line);
  exit(1);
}

const Token *Parser::peek(int lookahead) {
  if (index + lookahead < (int) tokens.size()) {
    return &tokens[index + lookahead];
  } else {
    return nullptr;
  }
}

template <unsigned int N>
const std::array<const Token *, N> Parser::peek() {
  std::array<const Token *, N> tokens;
  for (uint32_t i = 0; i < N; i++) tokens[i] = peek(i);
  return tokens;
}

optional<Token> Parser::pop(int lookahead) {
  if (index + lookahead < (int) tokens.size()) {
    Token t = tokens[index + lookahead];
    index += 1 + lookahead;
    return t;
  } else {
    return std::nullopt;
  }
}

template <unsigned int N>
std::array<optional<Token>, N> Parser::pop() {
  std::array<optional<Token>, N> tokens;
  for (uint32_t i = 0; i < N; i++) tokens[i] = pop();
  return tokens;
}

template <unsigned int N>
void Parser::expect(std::array<const Token *, N> &toks, std::array<Token::token_type, N> &expected) {
  for (uint32_t i = 0; i < N; i++) {
    if (toks[i] == nullptr) error("Unexpected EOF.", __LINE__);
    if (toks[i]->ty != expected[i]) error("Unexpected token: " + toks[i]->to_string(), __LINE__);
  }
}

vector<unique_ptr<Statement>> Parser::parse() {
  vector<unique_ptr<Statement>> statements;
  unique_ptr<Statement> s;
  Log::info("hmm\n");
  while (s = parse_statement()) {
    Log::info("is null = %d\n", s.get() == nullptr);
    statements.push_back(move(s));
  }

  return statements;
}

unique_ptr<Statement> Parser::parse_statement() {
  auto t = peek<2>();

  if (!t[0])
    return unique_ptr<Statement>();
  else if (!t[0])
    error("Dangling token " + t[0]->to_string(), __LINE__);
  Log::info("Got %s\n", t[0]->to_string().c_str());
  switch (t[0]->ty) {
    case Token::ID:
      if (t[1]->ty == Token::EQ)
        return parse_var_assignment();
      else if (t[1]->ty == Token::COLON)
        return parse_role_assignment();
      else
        return parse_connection();

    case Token::KW_MASTER:
    case Token::KW_MANAGERS:
    case Token::KW_ISLANDS:
    case Token::KW_WORKERS:
      Log::info("Here 0\n");
      if (t[1]->ty == Token::COLON)
        return parse_role_assignment();
      else
        return parse_connection();
    case Token::KW_TOPOLOGY:
      return parse_topology();
    case Token::CB_OPEN:
      return parse_compound_statement();
    case Token::KW_FOR:
      return parse_for();
    case Token::INT:
    case Token::KW_N_NODES:
    case Token::KW_PARTITION:
    case Token::LPAREN:
      return parse_connection();
    default:
      error("Unexpected token " + t[0]->to_string(), __LINE__);
  }

  // This should be unreachable
  exit(1);
  return unique_ptr<Statement>();
}

unique_ptr<Statement> Parser::parse_for() {
  auto t = peek<3>();
  std::array<Token::token_type, 3> exp{Token::KW_FOR, Token::ID, Token::KW_IN};
  expect<3>(t, exp);
  pop<3>();
  
  string id = t[1]->data;
  vector<unique_ptr<NodeRef>> refs = parse_abstract_node_ref_list();
  unique_ptr<Statement> st = parse_statement();

  Statement *for_st = (Statement *) new ForStatement(id, move(refs), move(st), t[0]->line, t[0]->column);
  return unique_ptr<Statement>(for_st);
}

unique_ptr<Statement> Parser::parse_compound_statement() {
  auto t = peek();
  if (t == nullptr)
    error("Unexpected eof\n", __LINE__);
  if (t->ty != Token::CB_OPEN)
    error("compound statement called without a leading curly brace\n", __LINE__);
  pop(); 
  vector<unique_ptr<Statement>> statements;
  while ((t = peek()) && t->ty != Token::CB_CLOSE)
    statements.push_back(parse_statement());

  if (t == nullptr)
    error("Unexpected eof\n", __LINE__);
  if (t->ty != Token::CB_CLOSE)
    error("This should be unreachable - what did you do??\n", __LINE__);
  pop();
  Statement *statement = (Statement *) new CompoundStatement(move(statements), t->line, t->column);
  return unique_ptr<Statement>(statement);
}

unique_ptr<Statement> Parser::parse_topology() {
  auto t = peek<3>();
  std::array<Token::token_type, 3> exp{Token::KW_TOPOLOGY, Token::ID, Token::COLON};
  expect<3>(t, exp);
  pop<3>();

  auto abs_ref_list = parse_abstract_node_ref_list();
  return unique_ptr<Statement>((Statement *) new TopologyStatement(TopologyStatement::get_topology_type(t[1]->data),
                                                                   move(abs_ref_list), t[0]->line, t[0]->column));
}

unique_ptr<Statement> Parser::parse_var_assignment() {
  auto t = peek<2>();
  std::array<Token::token_type, 2> exp{Token::ID, Token::EQ};
  expect<2>(t, exp);

  pop<2>();

  auto expr = parse_expr();

  return unique_ptr<Statement>((Statement *) new AssignmentStatement(t[0]->data, move(expr), t[0]->line, t[0]->column));
}

unique_ptr<Statement> Parser::parse_role_assignment() {
  Log::info("Here 1\n");
  auto t = peek<2>();

  if (t[0] == nullptr || t[1] == nullptr) error("Unexpected EOF", __LINE__);

  if (t[1]->ty != Token::COLON) error("Expected colon", __LINE__);
  pop<2>();
  switch (t[0]->ty) {
    case Token::KW_WORKERS:
    case Token::KW_MANAGERS:
    case Token::KW_ISLANDS: {
      vector<unique_ptr<NodeRef>> members = parse_node_ref_list();
      return unique_ptr<Statement>(
          (Statement *) new RoleStatement(move(members), Env::node_role_map.at(t[0]->ty), t[0]->line, t[0]->column));
    } break;
    case Token::KW_MASTER: {
      Log::info("Here 2\n");
      unique_ptr<Expr> exp = parse_expr();
      Log::info("Here 3\n");
      return unique_ptr<Statement>((Statement *) new RoleStatement(move(exp), t[0]->line, t[0]->column));
    } break;
    default:
      error("Unexpected token: " + t[0]->to_string(), __LINE__);
      // Unreachable
      exit(1);
      return unique_ptr<Statement>();
  }
}

unique_ptr<Statement> Parser::parse_connection() {
  Log::info("parse connection\n");
  auto start = peek();
  auto src = parse_abstract_node_ref_list();

  auto t = pop();
  if (t)
    Log::info((t->to_string() + "\n").c_str());
  if (t == std::nullopt || t->ty != Token::CONNECTION) error("Expected connection symbol", __LINE__);

  auto dst = parse_abstract_node_ref_list();

  auto p = new ConnectionStatement(move(src), move(dst), start->line, start->column);
  return unique_ptr<Statement>((Statement *) p);
}

vector<unique_ptr<NodeRef>> Parser::parse_abstract_node_ref_list() {
  vector<unique_ptr<NodeRef>> refs;

  refs.push_back(parse_abstract_node_ref());

  while (1) {
    auto tok = peek();
    if (tok == nullptr || tok->ty != Token::COMMA) break;

    pop();

    refs.push_back(parse_abstract_node_ref());
  }

  return refs;
}

vector<unique_ptr<NodeRef>> Parser::parse_node_ref_list() {
  vector<unique_ptr<NodeRef>> refs;

  refs.push_back(parse_node_ref());

  while (1) {
    auto tok = peek();
    if (tok == nullptr || tok->ty != Token::COMMA) break;

    pop();

    refs.push_back(parse_node_ref());
  }

  return refs;
}

unique_ptr<NodeRef> Parser::parse_abstract_node_ref() {
  auto tok = peek();
  if (tok == nullptr) error("Unexpected EOF.", __LINE__);

  switch (tok->ty) {
    case Token::KW_MASTER:
    case Token::KW_MANAGERS:
    case Token::KW_ISLANDS:
    case Token::KW_WORKERS:
      pop();
      return unique_ptr<NodeRef>((NodeRef *) new SpecialNodeRef(Env::node_role_map.at(tok->ty), tok->line, tok->column));
    default:
      return parse_node_ref();
  }
}

unique_ptr<NodeRef> Parser::parse_node_ref() {
  auto exp0 = parse_expr();
  int32_t line = exp0->line;
  int32_t column = exp0->column;

  auto tok = peek();
  if (tok == nullptr) error("Unexpected EOF.", __LINE__);

  if (tok->ty == Token::KW_UNTIL || tok->ty == Token::KW_THROUGH) {
    pop();
    auto exp1 = parse_expr();
    NodeRange *range = new NodeRange(move(exp0), move(exp1), tok->ty == Token::KW_THROUGH, line, column);
    return unique_ptr<NodeRef>((NodeRef *) range);
  } else {
    SingletonNode *singleton = new SingletonNode(move(exp0), line, column);
    return unique_ptr<NodeRef>((NodeRef *) singleton);
  }
}

unique_ptr<Expr> Parser::parse_expr() { return parse_sum_expr(); }

unique_ptr<Expr> Parser::parse_sum_expr() {
  auto lhs = parse_product_expr();
  const Token *tok;
  while ((tok = peek()) != nullptr) {
    switch (tok->ty) {
      case Token::ADD:
      case Token::SUB: {
        pop();
        auto op = tok->ty == Token::ADD ? ArithExpr::ADD : ArithExpr::SUB;
        auto rhs = parse_product_expr();
        int32_t line = lhs->line, column = lhs->column;
        auto exp = new ArithExpr(move(lhs), move(rhs), op, line, column);
        lhs = unique_ptr<Expr>((Expr *) exp);
      }
      default:
        goto done;
    }
  }
done:
  return lhs;
}

unique_ptr<Expr> Parser::parse_product_expr() {
  auto lhs = parse_partition_expr();
  const Token *tok;
  while ((tok = peek()) != nullptr) {
    Log::info("Stuck here ?\n");
    switch (tok->ty) {
      case Token::DIV:
      case Token::MUL: {
        pop();
        auto op = tok->ty == Token::DIV ? ArithExpr::DIV : (tok->ty == Token::MUL ? ArithExpr::MUL : ArithExpr::MOD);
        auto rhs = parse_partition_expr();
        int32_t line = lhs->line, column = lhs->column;
        auto exp = new ArithExpr(move(lhs), move(rhs), op, line, column);
        lhs = unique_ptr<Expr>((Expr *) exp);
      }
      default:
        goto done;
    }
  }
done:
  return lhs;
}

unique_ptr<Expr> Parser::parse_partition_expr() {
  auto tok = peek();
  if (tok == nullptr) error("Unexpected EOF", __LINE__);

  int32_t line = tok->line;
  int32_t column = tok->column;

  if (tok->ty != Token::KW_PARTITION) return parse_expr_inner();

  pop();

  auto partition_index = parse_expr();
  tok = peek();

  if (tok == nullptr || tok->ty != Token::KW_OF) error("Expected token of", __LINE__);

  pop();

  auto of_lower = parse_expr();
  auto range_ty = peek();
  if (range_ty == nullptr || (range_ty->ty != Token::KW_UNTIL && range_ty->ty != Token::KW_THROUGH))
    error("Unexpected token", __LINE__);
  pop();

  bool inclusive = Token::KW_THROUGH == range_ty->ty;

  auto of_upper = parse_expr();

  tok = peek();
  if (tok == nullptr || tok->ty != Token::KW_BY) error("Expected keyword by", __LINE__);

  pop();

  auto by = parse_expr();

  auto exp =
      new PartitionExpr(move(partition_index), move(of_lower), move(of_upper), inclusive, move(by), line, column);
  return unique_ptr<Expr>((Expr *) exp);
}

unique_ptr<Expr> Parser::parse_expr_inner() {
  auto tok = peek();
  Log::info("parse_expr_inner\n");
  if (tok == nullptr) error("Unexpected EOF", __LINE__);
  Log::info("tok %s\n", tok->to_string().c_str());
  Expr *e;
  switch (tok->ty) {
    case Token::ID:
      pop();
      e = (Expr *) new IdExpr(tok->data, tok->line, tok->column);
      break;

    case Token::INT:
      pop();
      Log::info("int %s\n", tok->data.c_str());
      e = (Expr *) new ConstExpr(std::stoi(tok->data), tok->line, tok->column);
      break;

    case Token::KW_MASTER:
      pop();
      e = (Expr *) new KWExpr(KWExpr::MASTER, tok->line, tok->column);
      break;

    case Token::KW_N_NODES:
      pop();
      e = (Expr *) new KWExpr(KWExpr::N_NODES, tok->line, tok->column);
      break;
    case Token::LPAREN: {
      pop();
      e = parse_sum_expr().release();
      auto x = pop();
      if (x == std::nullopt || x->ty != Token::RPAREN) error("Expected RPAREN", __LINE__);
      break;
    }
    default:
      e = nullptr;
      error("Unexpected token", __LINE__);
  }
  Log::info("What\n");
  return unique_ptr<Expr>(e);
}


///
/// AST
///

AST::AST(int32_t line, int32_t column) : line(line), column(column) {}
AST::~AST() {}


///
/// Expr
///

Expr::Expr(int32_t line, int32_t column) : AST(line, column) {}
Expr::~Expr() {}


///
/// ArithExpr
///

ArithExpr::ArithExpr(unique_ptr<Expr> l, unique_ptr<Expr> r, arith_op op, int32_t line, int32_t column)
    : Expr(line, column), l(move(l)), r(move(r)), op(op) {}

ArithExpr::~ArithExpr() {}

node_index_type ArithExpr::eval(Env &env) {
  node_index_type le = l->eval(env), re = r->eval(env);
  switch (op) {
    case MUL:
      return le * re;
    case DIV:
      return le / re;
    case MOD:
      return le % re;
    case ADD:
      return le + re;
    case SUB:
      return le - re;
  }
}

string ArithExpr::to_string() {
  string lhs = l->to_string(), rhs = r->to_string();
  string op;
  switch (this->op) {
    case MUL: op = "*"; break;
    case DIV: op = "/"; break;
    case MOD: op = "%"; break;
    case ADD: op = "+"; break;
    case SUB: op = "-"; break;
  }

  return lhs + " " + op + " " + rhs;
}


///
/// IdExpr
///

IdExpr::IdExpr(string id, int32_t line, int32_t column) : Expr(line, column), id(id) {}
IdExpr::~IdExpr() {}

node_index_type IdExpr::eval(Env &env) {
  auto it = env.vars.find(id);

  if (it != env.vars.end()) {
    return it->second;
  } else {
    return env.error("Variable with name " + id + " not found.");
  }
}

string IdExpr::to_string() { return id; }


///
/// PartitionExpr
///

PartitionExpr::PartitionExpr(unique_ptr<Expr> index, unique_ptr<Expr> lower, unique_ptr<Expr> upper, bool inclusive,
                             unique_ptr<Expr> divisor, int32_t line, int32_t column)
    : Expr(line, column),
      index(move(index)),
      lower(move(lower)),
      upper(move(upper)),
      divisor(move(divisor)),
      inclusive(inclusive) {}

PartitionExpr::~PartitionExpr() {}

node_index_type PartitionExpr::eval(Env &env) {
  node_index_type idx = index->eval(env);
  node_index_type lo = lower->eval(env);
  node_index_type hi = upper->eval(env);
  node_index_type div = divisor->eval(env);

  if (lo > hi)
    env.error("Lower bound of partition range should not be greater than the upper bound");

  if (inclusive)
    hi += 1;

  node_index_type dif = hi - lo;
  node_index_type psize = dif / div;
  node_index_type rem = dif % div;
  node_index_type pstart = psize * idx;
  if (pstart) {
    if (rem && idx < rem)
      pstart += rem - (rem - idx);
    else
      pstart += rem;
  }

  return std::min(lo + pstart, hi);
}

string PartitionExpr::to_string() {
  string index = this->index->to_string();
  string lo = this->lower->to_string();
  string hi = this->upper->to_string();
  string div = this->divisor->to_string();
  string range_connector = inclusive ? "through" : "until";
  return "partition " + index + " of " + lo + " " + range_connector + " " + hi + " by " + div;
}


///
/// KWExpr
///

KWExpr::KWExpr(KWExprTy ty, int32_t line, int32_t column) : Expr(line, column), ty(ty) {}

KWExpr::~KWExpr() {}

node_index_type KWExpr::eval(Env &env) {
  switch (ty) {
    case KWExpr::MASTER:
      return env.master;
    case KWExpr::N_NODES:
      return env.n_nodes;
  }
}

string KWExpr::to_string() {
  switch (ty) {
    case KWExpr::MASTER:
      return "masteR";
    case KWExpr::N_NODES:
      return "n_nodes";
  } 
}


///
/// ConstExpr
///

ConstExpr::ConstExpr(node_index_type value, int32_t line, int32_t column) 
  : Expr(line, column), value(value) {}

ConstExpr::~ConstExpr() {}

node_index_type ConstExpr::eval(Env &env) {
  return value;
}

string ConstExpr::to_string() {
  return std::to_string(value);
}


///
/// NodeRef
///

NodeRef::NodeRef(int32_t line, int32_t column) : AST(line, column) {}
NodeRef::~NodeRef() {}


///
/// NodeRange
///

NodeRange::NodeRange(unique_ptr<Expr> start, unique_ptr<Expr> end, bool inclusive, int32_t line, int32_t column)
  : NodeRef(line, column), start(move(start)), end(move(end)), inclusive(inclusive) {}
NodeRange::~NodeRange() {}

void NodeRange::get(vector<node_index_type> &indices, Env &env) {
  node_index_type lo = start->eval(env);
  node_index_type hi = end->eval(env) + (inclusive ? 1 : 0);

  for (; lo < hi; lo++)
    indices.push_back(lo);
}

string NodeRange::to_string() {
  string lo = start->to_string();
  string hi = end->to_string();
  string c = inclusive ? "through" : "until";

  return lo + " " + c + " " + hi;
}


///
/// SingletonNode
///

SingletonNode::SingletonNode(unique_ptr<Expr> node, int32_t line, int32_t column)
  : NodeRef(line, column), node(move(node)) {}

SingletonNode::~SingletonNode() {}

void SingletonNode::get(vector<node_index_type> &indices, Env &env) {
  indices.push_back(node->eval(env));
}

string SingletonNode::to_string() {
  return node->to_string();
}


///
/// SpecialNodeRef
///

SpecialNodeRef::SpecialNodeRef(node_role group, int32_t line, int32_t column)
  : NodeRef(line, column), group(group) {}

SpecialNodeRef::~SpecialNodeRef() {}

void SpecialNodeRef::get(vector<node_index_type> &indices, Env &env) {
  if (group == node_role::MASTER) {
    indices.push_back(env.master);
    return;
  }

  for (node_index_type i = 0; i < (node_index_type) env.node_roles.size(); i++)
    if (group == env.node_roles[i])
      indices.push_back(i);
}

string SpecialNodeRef::to_string() {
  return Env::node_role_string_map.at(group);
}


///
/// Statement
///

Statement::Statement(int32_t line, int32_t column) : AST(line, column) {}
Statement::~Statement() {}


///
/// ForStatement
///

ForStatement::ForStatement(string id, vector<unique_ptr<NodeRef>> refs, unique_ptr<Statement> statement, int32_t line, int32_t column)
  : Statement(line, column), id(move(id)), refs(move(refs)), statement(move(statement)) {}

ForStatement::~ForStatement() {}

void ForStatement::execute(Env &env) {
  vector<node_index_type> nodes;
  for (uint32_t i = 0; i < refs.size(); i++)
    refs[i]->get(nodes, env);

  for (auto it = nodes.begin(); it != nodes.end(); it++) {
    env.vars[id] = *it;
    statement->execute(env);
  }
}

string ForStatement::to_string() {
  string s;
  for (auto it = refs.begin(); it != refs.end(); it++)
    s += (*it)->to_string() + ", ";
  return "for " + id + " in " + s + statement->to_string();
}


///
/// CompoundStatement
///

CompoundStatement::CompoundStatement(vector<unique_ptr<Statement>> statements, int32_t line, int32_t column)
  : Statement(line, column), statements(move(statements)) {}

CompoundStatement::~CompoundStatement() {}

void CompoundStatement::execute(Env &env) {
  for (auto it = statements.begin(); it != statements.end(); it++)
    (*it)->execute(env);
}

string CompoundStatement::to_string() {
  string s = " {\n";
  for (auto it = statements.begin(); it != statements.end(); it++)
    s += (*it)->to_string() + "\n";
  s += "}";
  return s;
}


///
/// ConnectionStatement
///

ConnectionStatement::ConnectionStatement(vector<unique_ptr<NodeRef>> from, vector<unique_ptr<NodeRef>> to, int32_t line, int32_t column)
  : Statement(line, column), from(move(from)), to(move(to)) {}

ConnectionStatement::~ConnectionStatement() {}

void ConnectionStatement::execute(Env &env) {
  vector<node_index_type> from_nodes, to_nodes;
  
  for (uint32_t i = 0; i < from.size(); i++)
    from[i]->get(from_nodes, env);
  for (uint32_t i = 0; i < to.size(); i++)
    to[i]->get(to_nodes, env);

  for (uint32_t fi = 0; fi < from_nodes.size(); fi++)
    for (uint32_t ti = 0; ti < to_nodes.size(); ti++)
      env.connect(from_nodes[fi], to_nodes[ti]);
}

string ConnectionStatement::to_string() {
  vector<string> fs, ts;
  for (auto it = from.begin(); it != from.end(); it++)
    fs.push_back((*it)->to_string());
  for (auto it = to.begin(); it != to.end(); it++)
    ts.push_back((*it)->to_string());

  string fr = fs[0];
  for (auto it = ++fs.begin(); it != fs.end(); it++)
    fr = fr + ", " + *it;
  string t = ts[0];
  for (auto it = ++ts.begin(); it != ts.end(); it++)
    t = t + ", " + *it;

  return fr + " -> " + t;
}


///
/// RoleStatement
///

RoleStatement::RoleStatement(vector<unique_ptr<NodeRef>> members, node_role group, int32_t line, int32_t column)
  : Statement(line, column), members(move(members)), group(group) {}
RoleStatement::RoleStatement(unique_ptr<Expr> member, int32_t line, int32_t column)
  : Statement(line, column), group(node_role::MASTER) {
  NodeRef *ref = (NodeRef *) new SingletonNode(move(member), line, column);
  members.emplace_back(ref);
}

RoleStatement::~RoleStatement() {}

void RoleStatement::execute(Env &env) {
  vector<node_index_type> member_nodes;
  for (uint32_t i = 0; i < members.size(); i++)
    members[i]->get(member_nodes, env);

  for (uint32_t i = 0; i < member_nodes.size(); i++)
    env.node_roles[member_nodes[i]] = group;
}

string RoleStatement::to_string() {
   vector<string> fs;
  for (auto it = members.begin(); it != members.end(); it++)
    fs.push_back((*it)->to_string());

  string fr = fs[0];
  for (auto it = ++fs.begin(); it != fs.end(); it++)
    fr = fr + ", " + *it;

  return Env::node_role_string_map.at(group) + ": " + fr;
}


///
/// TopologyStatement
///

TopologyStatement::TopologyStatement(topology_type topology, vector<unique_ptr<NodeRef>> members, int32_t line, int32_t column) 
  : Statement(line, column), topology(topology), members(move(members)) {}

TopologyStatement::~TopologyStatement() {}

TopologyStatement::topology_type TopologyStatement::get_topology_type(string str) {
  if (str == "ring")
    return TopologyStatement::RING;
  else {
    Log::fatal("Invalid topology type %s\n", str.c_str());
    exit(1);
  }
}

void TopologyStatement::execute(Env &env) {
  vector<node_index_type> member_nodes;
  for (uint32_t i = 0; i < members.size(); i++)
    members[i]->get(member_nodes, env);

  set<node_index_type> node_set(member_nodes.begin(), member_nodes.end());

  switch (topology) {
    case RING: {
      if (node_set.size() <= 1)
        return;
      for (auto it = node_set.cbegin(); it != --node_set.cend();) {
        node_index_type a = *it;
        node_index_type b = *++it;
        env.connect(a, b);
        env.connect(b, a);
      }

      node_index_type first = *node_set.cbegin();
      node_index_type last = *--node_set.cend();
      env.connect(first, last);
      env.connect(last, first);
      break;
    }
  }
}

string TopologyStatement::to_string() {
  vector<string> fs;
  for (auto it = members.begin(); it != members.end(); it++)
    fs.push_back((*it)->to_string());

  string fr = fs[0];
  for (auto it = ++fs.begin(); it != fs.end(); it++)
    fr = fr + ", " + *it;
  
  string top;
  switch (topology) {
    case RING:
      top = "ring";
      break;
  }

  return "topology " + top + ": " + fr;
}


///
/// AssignmentStatement
///

AssignmentStatement::AssignmentStatement(string id, unique_ptr<Expr> value, int32_t line, int32_t column)
  : Statement(line, column), id(id), value(move(value)) {}

AssignmentStatement::~AssignmentStatement() {}

void AssignmentStatement::execute(Env &env) {
  node_index_type val = value->eval(env);
  env.vars[id] = val;
}

string AssignmentStatement::to_string() {
  return id + " = " + value->to_string();
}

const unordered_map<Token::token_type, node_role> Env::node_role_map = {
  {Token::KW_MASTER, MASTER},
  {Token::KW_ISLANDS, ISLANDS},
  {Token::KW_MANAGERS, MANAGERS},
  {Token::KW_WORKERS, WORKERS}
};

const unordered_map<node_role, string> Env::node_role_string_map = {
  {node_role::MASTER, "master"},
  {node_role::ISLANDS, "islands"},
  {node_role::MANAGERS, "managers"},
  {node_role::WORKERS, "workers"}
};

node_index_type Env::error(string message) {
  Log::fatal("Encountered error %s\n", message.c_str());
  exit(1);
}

Env::Env(node_index_type n_nodes) 
  : node_roles(vector<node_role>(n_nodes, node_role::WORKERS)),
    connections(vector<vector<bool>>(n_nodes, vector<bool>(n_nodes, false))), master(0), n_nodes(n_nodes) { }

void Env::connect(node_index_type from, node_index_type to) {
  Log::info("Connecting %d -> %d\n", (int)from, (int)to);
  connections[from][to] = true;
}


///
/// ArchipelagoConfig
///

ArchipelagoConfig::ArchipelagoConfig(node_index_type master_id, vector<vector<bool>> connections, vector<node_role> node_roles) :
  master_id(master_id),  connections(move(connections)), node_roles(node_roles) {}

ArchipelagoConfig ArchipelagoConfig::from_string(string str, int32_t n_nodes, map<string, node_index_type> &define_map) {

  string tmp_file = std::tmpnam(nullptr);
  // The pre processor provided by clang nas no space between the -o and
  // the tmp_file path. There is one with gcc cpp.
#if defined(__clang__)
  string command = "cpp -P -o" + tmp_file;
#else
  string command = "cpp -P -o " + tmp_file;
#endif
  for (auto it = define_map.begin(); it != define_map.end(); it++)
    command = command + " -D" + it->first + "=" + std::to_string(it->second);
  Log::info("cpp command: %s\n", command.c_str());

  FILE *pre_processor = popen(command.c_str(), "w");
  if (!pre_processor) {
    Log::info("Null pre_process\n");
  }

  size_t total_wrote = 0;
  while (1) {
    int n = fprintf(pre_processor, "%s", str.c_str() + total_wrote);
    if (n < 0) {
      Log::info("Encountered error writing cpp\n");
      pclose(pre_processor);
      exit(1);
    }
    total_wrote += n;
    if (total_wrote >= str.size())
      break;
  }
  
  pclose(pre_processor);

  std::ifstream t(tmp_file);
  std::stringstream buf;
  buf << t.rdbuf();

  str = buf.str();
  Log::info("\n%s\n", str.c_str());
  Tokenizer tokenizer(str);

  auto tokens = tokenizer.tokenize();
  for (auto it = tokens.begin(); it != tokens.end(); it++)
    Log::info("%s\n", it->debug().c_str());
  auto parser = Parser(tokens);
  auto statements = parser.parse();
  for (auto st = statements.begin(); st != statements.end(); st++)
    std::cout << (*st)->to_string() << "\n";
  Env env(n_nodes);
  for (auto st = statements.begin(); st != statements.end(); st++)
    (*st)->execute(env);

  return ArchipelagoConfig(env.master, env.connections, env.node_roles);
}

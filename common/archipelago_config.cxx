#include "archipelago_config.hxx"

#include <utility>
using std::move;

#include <iostream>
#include <sstream>

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
    {"until",     Token::KW_UNTIL    }
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
    if (toks[i] == nullptr) error("Unexpected EOF.");
    if (toks[i]->ty != expected[i]) error("Unexpected token: " + toks[i]->to_string());
  }
}

vector<unique_ptr<Statement>> Parser::parse() {
  vector<unique_ptr<Statement>> statements;
  unique_ptr<Statement> s;
  while (s = parse_statement()) statements.push_back(move(s));

  return statements;
}

unique_ptr<Statement> Parser::parse_statement() {
  auto t = peek<2>();

  if (!t[0])
    return unique_ptr<Statement>();
  else if (!t[0])
    error("Dangling token " + t[0]->to_string());

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
      if (t[1]->ty == Token::COLON)
        return parse_role_assignment();
      else
        return parse_connection();
    case Token::INT:
    case Token::KW_N_NODES:
    case Token::KW_PARTITION:
    case Token::LPAREN:
      return parse_connection();
    default:
      error("Unexpected token " + t[0]->to_string());
  }

  // This should be unreachable
  exit(1);
  return unique_ptr<Statement>();
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
  auto t = peek<2>();

  if (t[0] == nullptr || t[1] == nullptr) error("Unexpected EOF");

  if (t[1]->ty != Token::COLON) error("Expected colon");

  switch (t[0]->ty) {
    case Token::KW_WORKERS:
    case Token::KW_MANAGERS:
    case Token::KW_ISLANDS: {
      vector<unique_ptr<NodeRef>> members = parse_node_ref_list();
      return unique_ptr<Statement>(
          (Statement *) new RoleStatement(move(members), special_group_map.at(t[0]->ty), t[0]->line, t[0]->column));
    } break;
    case Token::KW_MASTER: {
      unique_ptr<Expr> exp = parse_expr();
      return unique_ptr<Statement>((Statement *) new RoleStatement(move(exp), t[0]->line, t[0]->column));
    } break;
    default:
      error("Unexpected token: " + t[0]->to_string());
      // Unreachable
      exit(1);
      return unique_ptr<Statement>();
  }
}

unique_ptr<Statement> Parser::parse_connection() {
  auto start = peek();
  auto src = parse_abstract_node_ref_list();

  auto t = pop();
  if (t == std::nullopt || t->ty != Token::CONNECTION) error("Expected connection symbol.");

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
  if (tok == nullptr) error("Unexpected EOF.");

  switch (tok->ty) {
    case Token::KW_MASTER:
    case Token::KW_MANAGERS:
    case Token::KW_ISLANDS:
    case Token::KW_WORKERS:
      return unique_ptr<NodeRef>((NodeRef *) new SpecialNodeRef(special_group_map[tok->ty], tok->line, tok->column));
    default:
      return parse_node_ref();
  }
}

unique_ptr<NodeRef> Parser::parse_node_ref() {
  auto exp0 = parse_expr();
  int32_t line = exp0->line;
  int32_t column = exp0->column;

  auto tok = pop();
  if (tok == std::nullopt) error("Unexpected EOF.");

  if (tok->ty == Token::KW_UNTIL || tok->ty == Token::KW_THROUGH) {
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
        auto op = tok->ty == Token::ADD ? ArithExpr::ADD : ArithExpr::SUB;
        auto rhs = parse_product_expr();
        int32_t line = lhs->line, column = lhs->column;
        auto exp = new ArithExpr(move(lhs), move(rhs), op, line, column);
        lhs = unique_ptr<Expr>((Expr *) exp);
      }
      default:
        break;
    }
  }
  return lhs;
}

unique_ptr<Expr> Parser::parse_product_expr() {
  auto lhs = parse_partition_expr();
  const Token *tok;
  while ((tok = peek()) != nullptr) {
    switch (tok->ty) {
      case Token::DIV:
      case Token::MUL: {
        auto op = tok->ty == Token::DIV ? ArithExpr::DIV : (tok->ty == Token::MUL ? ArithExpr::MUL : ArithExpr::MOD);
        auto rhs = parse_partition_expr();
        int32_t line = lhs->line, column = lhs->column;
        auto exp = new ArithExpr(move(lhs), move(rhs), op, line, column);
        lhs = unique_ptr<Expr>((Expr *) exp);
      }
      default:
        break;
    }
  }
  return lhs;
}

unique_ptr<Expr> Parser::parse_partition_expr() {
  auto tok = peek();
  if (tok == nullptr) error("Unexpected EOF");

  int32_t line = tok->line;
  int32_t column = tok->column;

  if (tok->ty != Token::KW_PARTITION) return parse_expr_inner();

  pop();

  auto partition_index = parse_expr();
  tok = peek();

  if (tok == nullptr || tok->ty != Token::KW_OF) error("Expected token of");

  pop();

  auto of_lower = parse_expr();
  auto range_ty = peek();
  if (range_ty == nullptr || (range_ty->ty != Token::KW_UNTIL && range_ty->ty != Token::KW_THROUGH))
    error("Unexpected token");
  pop();

  bool inclusive = Token::KW_THROUGH == range_ty->ty;

  auto of_upper = parse_expr();

  tok = peek();
  if (tok == nullptr || tok->ty != Token::KW_BY) error("Expected keyword by");

  pop();

  auto by = parse_expr();

  auto exp =
      new PartitionExpr(move(partition_index), move(of_lower), move(of_upper), inclusive, move(by), line, column);
  return unique_ptr<Expr>((Expr *) exp);
}

unique_ptr<Expr> Parser::parse_expr_inner() {
  auto tok = peek();

  if (tok == nullptr) error("Unexpected EOF");

  Expr *e;
  switch (tok->ty) {
    case Token::ID:
      pop();
      e = (Expr *) new IdExpr(tok->data, tok->line, tok->column);
      break;

    case Token::INT:
      pop();
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
      if (x == std::nullopt || x->ty != Token::RPAREN) error("Expected RPAREN");
      break;
    }
    default:
      e = nullptr;
      error("Unexpected token");
  }

  return unique_ptr<Expr>(e);
}

AST::AST(int32_t line, int32_t column) : line(line), column(column) {}

Expr::Expr(int32_t line, int32_t column) : AST(line, column) {}

ArithExpr::ArithExpr(unique_ptr<Expr> l, unique_ptr<Expr> r, arith_op op, int32_t line, int32_t column)
    : Expr(line, column), l(move(l)), r(move(r)), op(op) {}

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

IdExpr::IdExpr(string id, int32_t line, int32_t column) : Expr(line, column), id(id) {}

node_index_type IdExpr::eval(Env &env) {
  auto it = env.vars.find(id);

  if (it != env.vars.end()) {
    return it->second;
  } else {
    return env.error("Variable with name " + id + " not found.");
  }
}

PartitionExpr::PartitionExpr(unique_ptr<Expr> index, unique_ptr<Expr> lower, unique_ptr<Expr> upper, bool inclusive,
                             unique_ptr<Expr> divisor, int32_t line, int32_t column)
    : Expr(line, column),
      index(move(index)),
      lower(move(lower)),
      upper(move(upper)),
      divisor(move(divisor)),
      inclusive(inclusive) {}

node_index_type PartitionExpr::eval(Env &env) {
  // TODO
  return 0;
}

KWExpr::KWExpr(KWExprTy ty, int32_t line, int32_t column) : Expr(line, column), ty(ty) {}
node_index_type KWExpr::eval(Env &env) {
  
}

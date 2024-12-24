#include "CPP.h"
#include <cctype>
#include <unordered_set>
#include <sstream>
#include <iostream>
#include "utils.h"


// Wrapper function for compatibility
inline size_t FindClose(const std::vector<CPP::Tkn>& tokens, size_t start, char open, char close) {
    return ::FindClose(tokens, start, open, close, [](const CPP::Tkn& token) { return token.val[0]; });
}

// Constructor
CPP::CPP() {}

// Tokenize source code
std::vector<CPP::Tkn> CPP::Tknz(const std::string& code) {
    std::vector<Tkn> tkns;
    std::ostringstream cur;

    for (char c : code) {
        if (std::isspace(c)) {
            if (!cur.str().empty()) {
                tkns.push_back(Classify(cur.str()));
                cur.str("");
                cur.clear();
            }
        } else if (std::ispunct(c) && c != '_') {
            if (!cur.str().empty()) {
                tkns.push_back(Classify(cur.str()));
                cur.str("");
                cur.clear();
            }
            tkns.push_back({Tkn::Sym, std::string(1, c)});
        } else {
            cur << c;
        }
    }

    if (!cur.str().empty()) {
        tkns.push_back(Classify(cur.str()));
    }

    return tkns;
}

// Classify tokens
CPP::Tkn CPP::Classify(const std::string& tkn) {
    static const std::unordered_set<std::string> kws = {
        "int", "float", "double", "char", "if", "else", "for", "while", "return",
        "void", "class", "struct", "enum", "template", "namespace", "constexpr",
        "concept", "co_yield", "co_return", "co_await", "inline", "virtual",
        "override", "final", "export", "module", "auto", "static", "public",
        "protected", "private"
    };

    if (kws.count(tkn)) return {Tkn::Kw, tkn};
    if (std::isdigit(tkn[0])) return {Tkn::Lit, tkn};
    return {Tkn::Id, tkn};
}

// Parse class construct
CPP::Nd CPP::Cls(const std::vector<Tkn>& tkns, size_t& index) {
    Nd node = {"class", tkns[index + 1].val, ""};
    index += 2; // Skip "class" and its name
    if (tkns[index].val == "{") {
        size_t closeIdx = FindClose(tkns, index, '{', '}');
        for (size_t i = index + 1; i < closeIdx; ++i) {
            if (tkns[i].type == Tkn::Kw && tkns[i].val == "void") {
                node.children.push_back(Func(tkns, i));
            }
        }
        index = closeIdx;
    }
    return node;
}

// Parse struct construct
CPP::Nd CPP::Strct(const std::vector<Tkn>& tkns, size_t& index) {
    Nd node = {"struct", tkns[index + 1].val, ""};
    index += 2;
    if (tkns[index].val == "{") {
        size_t closeIdx = FindClose(tkns, index, '{', '}');
        for (size_t i = index + 1; i < closeIdx; ++i) {
            if (tkns[i].type == Tkn::Kw && tkns[i].val == "int") {
                node.children.push_back({"var", tkns[i + 1].val, "int"});
                ++i;
            }
        }
        index = closeIdx;
    }
    return node;
}

// Parse function construct
CPP::Nd CPP::Func(const std::vector<Tkn>& tkns, size_t& index) {
    Nd node = {"func", tkns[index + 1].val, tkns[index].val};
    index += 2;
    if (tkns[index].val == "(") {
        size_t closeIdx = FindClose(tkns, index, '(', ')');
        index = closeIdx;
    }
    return node;
}

// Parse enum construct
CPP::Nd CPP::Enm(const std::vector<Tkn>& tkns, size_t& index) {
    Nd node = {"enum", tkns[index + 1].val, ""};
    index += 2;
    if (tkns[index].val == "{") {
        size_t closeIdx = FindClose(tkns, index, '{', '}');
        for (size_t i = index + 1; i < closeIdx; ++i) {
            if (tkns[i].type == Tkn::Id) {
                node.children.push_back({"value", tkns[i].val, ""});
            }
        }
        index = closeIdx;
    }
    return node;
}

// Parse template construct
CPP::Nd CPP::Tmplt(const std::vector<Tkn>& tkns, size_t& index) {
    Nd node = {"template", "", ""};
    index += 1;
    if (tkns[index].val == "<") {
        size_t closeIdx = FindClose(tkns, index, '<', '>');
        node.value = tkns[index + 1].val; // Assume a single type parameter
        index = closeIdx;
    }
    return node;
}

// Parse namespace construct
CPP::Nd CPP::Nsp(const std::vector<Tkn>& tkns, size_t& index) {
    Nd node = {"namespace", tkns[index + 1].val, ""};
    index += 2;
    if (tkns[index].val == "{") {
        size_t closeIdx = FindClose(tkns, index, '{', '}');
        index = closeIdx;
    }
    return node;
}

// Parse macro construct
CPP::Nd CPP::Mcr(const std::vector<Tkn>& tkns, size_t& index) {
    Nd node = {"macro", tkns[index].val, ""};
    ++index;
    return node;
}

// Parse coroutine construct
CPP::Nd CPP::Co(const std::vector<Tkn>& tkns, size_t& index) {
    Nd node = {"coroutine", tkns[index + 1].val, ""};
    index += 2;
    return node;
}


#include "CPP.h"
#include <cctype>
#include <algorithm>
#include <iostream>

/**
 * Constructor for the CPP class. Initializes the parser.
 */
CPP::CPP() {}

/**
 * Tokenizes the source code into a sequence of meaningful tokens.
 * @param code The C++ source code to tokenize.
 * @return A vector of tokens.
 */
std::vector<CPP::Tkn> CPP::Tknz(const std::string& code) {
    std::vector<Tkn> tkns;
    std::string cur;

    for (size_t i = 0; i < code.size(); ++i) {
        char c = code[i];
        if (std::isspace(c)) {
            if (!cur.empty()) {
                tkns.push_back(Classify(cur));
                cur.clear();
            }
        } else if (std::ispunct(c) && c != '_') {
            if (!cur.empty()) {
                tkns.push_back(Classify(cur));
                cur.clear();
            }
            tkns.push_back({Tkn::Sym, std::string(1, c)});
        } else {
            cur += c;
        }
    }

    if (!cur.empty()) {
        tkns.push_back(Classify(cur));
    }

    return tkns;
}

/**
 * Classifies a string into a specific token type.
 * @param tkn The string to classify.
 * @return The classified token.
 */
CPP::Tkn CPP::Classify(const std::string& tkn) {
    static const std::vector<std::string> kws = {
        "int", "float", "double", "char", "if", "else", "for", "while", "return",
        "void", "class", "struct", "enum", "template", "namespace", "constexpr",
        "concept", "co_yield", "co_return", "co_await", "inline", "virtual",
        "override", "final", "constexpr", "export", "module"
    };

    if (std::find(kws.begin(), kws.end(), tkn) != kws.end()) {
        return {Tkn::Kw, tkn};
    } else if (std::isdigit(tkn[0])) {
        return {Tkn::Lit, tkn};
    } else {
        return {Tkn::Id, tkn};
    }
}

/**
 * Parses the provided C++ source code into a structured tree of nodes.
 * @param code The C++ source code to parse.
 * @return The root node of the parsed structure.
 */
CPP::Nd CPP::Prs(const std::string& code) {
    auto t = Tknz(code);
    Nd root = {"root", "", ""};

    for (size_t i = 0; i < t.size(); ++i) {
        if (t[i].type == Tkn::Kw) {
            if (t[i].val == "class") {
                root.ch.push_back(Cls(t, i));
            } else if (t[i].val == "struct") {
                root.ch.push_back(Strct(t, i));
            } else if (t[i].val == "enum") {
                root.ch.push_back(Enm(t, i));
            } else if (t[i].val == "template") {
                root.ch.push_back(Tmplt(t, i));
            } else if (t[i].val == "constexpr") {
                root.ch.push_back(Cxpr(t, i));
            } else if (t[i].val == "concept") {
                root.ch.push_back(Cpt(t, i));
            } else if (t[i].val == "namespace") {
                root.ch.push_back(Nsp(t, i));
            } else if (t[i].val == "module") {
                root.ch.push_back({"module", t[i + 1].val, ""});
                i++;
            }
        } else if (t[i].val[0] == '#') {
            root.ch.push_back(Mcr(t, i));
        }
    }

    return root;
}

// Implement remaining parsing methods (Cls, Strct, etc.) similarly.



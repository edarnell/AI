#ifndef CPP_H
#define CPP_H

#include <string>
#include <vector>

/**
 * @class CPP
 * The CPP class is responsible for parsing C++ source code into a structured
 * logical representation. It provides methods to tokenize input, parse constructs,
 * and extract logical relationships between constructs.
 */
class CPP {
public:
    /**
     * Token represents a meaningful unit of C++ code, such as keywords,
     * identifiers, symbols, literals, or unknown text.
     */
    struct Tkn {
        enum TknType { Kw, Id, Sym, Lit, Unk } type; // Token types: Keyword, Identifier, Symbol, Literal, Unknown
        std::string val; // The actual string value of the token
    };

    /**
     * Node represents a parsed construct in the source code, such as a function,
     * class, variable, or other meaningful entity.
     */
    struct Nd {
        std::string typ;    // The type of the construct (e.g., func, cls, var)
        std::string nm;     // The name of the construct
        std::string val;    // Additional details (e.g., type, value, return type)
        std::vector<Nd> ch; // Child nodes representing nested constructs
    };

    /**
     * Relationship defines logical links between constructs, such as parent-child,
     * references, or scoping relationships.
     */
    struct Rel {
        std::string p;   // Parent construct
        std::string c;   // Child construct
        std::string typ; // Type of relationship (e.g., parent-child, reference, scope)
    };

    CPP(); // Constructor initializes the parser

    /**
     * Parses the provided C++ source code into a structured tree of nodes.
     * @param code The C++ source code to parse.
     * @return The root node of the parsed structure.
     */
    Nd Prs(const std::string& code);

    /**
     * Extracts logical relationships between parsed constructs.
     * @param root The root node of the parsed tree.
     * @return A list of relationships between constructs.
     */
    std::vector<Rel> ExtRel(const Nd& root);

private:
    /**
     * Tokenizes the source code into a sequence of meaningful tokens.
     * @param code The C++ source code to tokenize.
     * @return A vector of tokens.
     */
    std::vector<Tkn> Tknz(const std::string& code);

    // Parsing methods for specific constructs
    Nd Func(const std::vector<Tkn>& t, size_t& i);    // Parses a function construct
    Nd Cls(const std::vector<Tkn>& t, size_t& i);     // Parses a class construct
    Nd Strct(const std::vector<Tkn>& t, size_t& i);   // Parses a struct construct
    Nd Enm(const std::vector<Tkn>& t, size_t& i);     // Parses an enum construct
    Nd Tmplt(const std::vector<Tkn>& t, size_t& i);   // Parses a template construct
    Nd Lmb(const std::vector<Tkn>& t, size_t& i);     // Parses a lambda construct
    Nd Cxpr(const std::vector<Tkn>& t, size_t& i);    // Parses a constexpr construct
    Nd Cpt(const std::vector<Tkn>& t, size_t& i);     // Parses a concept construct
    Nd Attrib(const std::vector<Tkn>& t, size_t& i);  // Parses an attribute construct
    Nd Nsp(const std::vector<Tkn>& t, size_t& i);     // Parses a namespace construct
    Nd Mcr(const std::vector<Tkn>& t, size_t& i);     // Parses a macro construct
    Nd Co(const std::vector<Tkn>& t, size_t& i);      // Parses a coroutine construct

    /**
     * Finds the closing bracket corresponding to an opening bracket.
     * @param t The sequence of tokens.
     * @param start The index of the opening bracket.
     * @param open The opening bracket character.
     * @param close The closing bracket character.
     * @return The index of the closing bracket.
     */
    size_t FindClose(const std::vector<Tkn>& t, size_t start, char open, char close);
};

#endif // CPP_H



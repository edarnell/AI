#ifndef CPP_H
#define CPP_H

#include <string>
#include <vector>

/**
 * @class CPP
 * Parses C++ source code into a structured logical representation.
 */
class CPP {
public:
    /**
     * @struct Tkn
     * Represents a meaningful unit of C++ code.
     */
    struct Tkn {
        enum TknType { Kw, Id, Sym, Lit, Unk } type; // Token types: Keyword, Identifier, Symbol, Literal, Unknown
        std::string val;                             // The actual token value
    };

    /**
     * @struct Nd
     * Represents a parsed construct in the source code.
     */
    struct Nd {
        std::string type;             // Type of construct (e.g., class, func, var)
        std::string name;             // Name of the construct
        std::string value;            // Additional details (e.g., type, value)
        std::vector<Nd> children;     // Nested constructs
    };

    CPP(); // Constructor initializes the parser

    /**
     * Tokenizes the source code into meaningful units.
     * @param code The C++ source code.
     * @return A vector of tokens.
     */
    std::vector<Tkn> Tknz(const std::string& code);

    /**
     * Parses the source code into a structured tree of nodes.
     * @param code The C++ source code.
     * @return The root node of the parsed structure.
     */
    Nd Prs(const std::string& code);

private:
    /**
     * Classifies a string into a specific token type.
     * @param tkn The string to classify.
     * @return The classified token.
     */
    Tkn Classify(const std::string& tkn);

    /**
     * Finds the matching closing bracket for a given opening bracket.
     * @param tokens The sequence of tokens.
     * @param start The index of the opening bracket.
     * @param open The opening bracket character.
     * @param close The closing bracket character.
     * @return The index of the closing bracket.
     */
    size_t FindClose(const std::vector<Tkn>& tokens, size_t start, char open, char close);

    // Parsing methods for specific constructs
    Nd Cls(const std::vector<Tkn>& tokens, size_t& index);      // Parses a class construct
    Nd Strct(const std::vector<Tkn>& tokens, size_t& index);    // Parses a struct construct
    Nd Func(const std::vector<Tkn>& tokens, size_t& index);     // Parses a function construct
    Nd Enm(const std::vector<Tkn>& tokens, size_t& index);      // Parses an enum construct
    Nd Tmplt(const std::vector<Tkn>& tokens, size_t& index);    // Parses a template construct
    Nd Nsp(const std::vector<Tkn>& tokens, size_t& index);      // Parses a namespace construct
    Nd Mcr(const std::vector<Tkn>& tokens, size_t& index);      // Parses a macro construct
    Nd Co(const std::vector<Tkn>& tokens, size_t& index);       // Parses a coroutine construct
};

#endif // CPP_H



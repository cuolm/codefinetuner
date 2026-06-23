using TreeSitter;
using Xunit;

namespace TreeSitterIec61131_3St.Tests;

public class LanguageTests
{
    // A small but complete ST snippet used throughout the tests.
    private const string SampleCode = """
        PROGRAM Counter
        VAR
            count : INT := 0;
        END_VAR
            count := count + 1;
            Add(count, 1);
        END_PROGRAM

        FUNCTION Add : INT
        VAR_INPUT
            a : INT;
            b : INT;
        END_VAR
            Add := a + b;
        END_FUNCTION
        """;

    // ── Grammar loading ──────────────────────────────────────────────────────

    [Fact]
    public void CanLoadGrammar()
    {
        using var language = Language.Create();
        using var parser = new Parser(language);
        Assert.NotNull(parser);
    }

    [Fact]
    public void ParsesCodeWithoutErrors()
    {
        using var language = Language.Create();
        using var parser = new Parser(language);
        using var tree = parser.Parse(SampleCode);
        Assert.False(tree.RootNode.HasError);
    }

    // ── Embedded queries ─────────────────────────────────────────────────────

    [Fact]
    public void HighlightsQueryIsAvailable()
    {
        Assert.NotNull(Language.HighlightsQuery);
        Assert.NotEmpty(Language.HighlightsQuery);
    }

    [Fact]
    public void TagsQueryIsAvailable()
    {
        Assert.NotNull(Language.TagsQuery);
        Assert.NotEmpty(Language.TagsQuery);
    }

    // ── Custom queries ───────────────────────────────────────────────────────
    //
    // Queries use the S-expression pattern language built into tree-sitter.
    // The general shape is:
    //
    //   (node_type field: (child_type) @capture_name)
    //
    // Use Language.Create() to get a Language, then:
    //
    //   using var query  = new Query(language, "<pattern>");
    //   using var cursor = query.Execute(tree.RootNode);
    //   foreach (var capture in cursor.Captures) { ... }
    //
    // capture.Name   → the @capture label from the pattern
    // capture.Node   → the matched syntax node
    // capture.Node.Text → the raw source text of that node

    [Fact]
    public void CustomQuery_FindProgramNames()
    {
        using var language = Language.Create();
        using var parser = new Parser(language);
        using var tree = parser.Parse(SampleCode);

        using var query = new Query(language,
            "(program_declaration name: (identifier) @name)");
        using var cursor = query.Execute(tree.RootNode);

        var names = cursor.Captures.Select(c => c.Node.Text).ToList();

        Assert.Equal(["Counter"], names);
    }

    [Fact]
    public void CustomQuery_FindFunctionNames()
    {
        using var language = Language.Create();
        using var parser = new Parser(language);
        using var tree = parser.Parse(SampleCode);

        using var query = new Query(language,
            "(function_declaration name: (identifier) @name)");
        using var cursor = query.Execute(tree.RootNode);

        var names = cursor.Captures.Select(c => c.Node.Text).ToList();

        Assert.Equal(["Add"], names);
    }

    [Fact]
    public void CustomQuery_FindInputVariableNames()
    {
        using var language = Language.Create();
        using var parser = new Parser(language);
        using var tree = parser.Parse(SampleCode);

        // Scope the variable search to VAR_INPUT blocks only
        using var query = new Query(language,
            "(var_input (variable_declaration names: (identifier) @name))");
        using var cursor = query.Execute(tree.RootNode);

        var names = cursor.Captures.Select(c => c.Node.Text).ToList();

        Assert.Equal(["a", "b"], names);
    }

    [Fact]
    public void CustomQuery_FindFunctionCalls()
    {
        using var language = Language.Create();
        using var parser = new Parser(language);
        using var tree = parser.Parse(SampleCode);

        using var query = new Query(language,
            "(call_expression function: (identifier) @callee)");
        using var cursor = query.Execute(tree.RootNode);

        var callees = cursor.Captures.Select(c => c.Node.Text).ToList();

        Assert.Contains("Add", callees);
    }

    [Fact]
    public void CustomQuery_CombinePatternsWithMultipleCaptures()
    {
        using var language = Language.Create();
        using var parser = new Parser(language);
        using var tree = parser.Parse(SampleCode);

        // A single query can contain multiple patterns; each capture gets a
        // distinct @name that you can filter on via capture.Name.
        using var query = new Query(language, """
            (program_declaration  name: (identifier) @pou.name)
            (function_declaration name: (identifier) @pou.name)
            """);
        using var cursor = query.Execute(tree.RootNode);

        var names = cursor.Captures
            .Where(c => c.Name == "pou.name")
            .Select(c => c.Node.Text)
            .ToList();

        Assert.Equal(["Counter", "Add"], names);
    }
}

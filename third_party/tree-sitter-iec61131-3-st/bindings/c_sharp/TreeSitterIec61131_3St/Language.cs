using System.Reflection;
using System.Runtime.InteropServices;
using TreeSitter;

namespace TreeSitterIec61131_3St;

/// <summary>
/// Provides access to the IEC 61131-3 Structured Text tree-sitter grammar.
/// </summary>
/// <remarks>
/// The native library <c>libtree-sitter-iec61131-3-st</c> must be available at
/// runtime (e.g. built with <c>make</c> and installed, or present on
/// <c>LD_LIBRARY_PATH</c> / <c>PATH</c>).
/// </remarks>
public static class Language
{
    private static string NativeLibraryName =>
        RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "tree-sitter-iec61131-3-st.dll"
            : RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                ? "libtree-sitter-iec61131-3-st.dylib"
                : "libtree-sitter-iec61131-3-st.so";

    private const string FunctionName = "tree_sitter_iec61131_3_st";

    /// <summary>
    /// Creates a <see cref="TreeSitter.Language"/> instance for the
    /// IEC 61131-3 Structured Text grammar.
    /// </summary>
    public static TreeSitter.Language Create()
        => new TreeSitter.Language(NativeLibraryName, FunctionName);

    /// <summary>The syntax highlighting query for this grammar, or <see langword="null"/> if not embedded.</summary>
    public static string? HighlightsQuery => GetQuery("highlights.scm");

    /// <summary>The language injection query for this grammar, or <see langword="null"/> if not embedded.</summary>
    public static string? InjectionsQuery => GetQuery("injections.scm");

    /// <summary>The local variable query for this grammar, or <see langword="null"/> if not embedded.</summary>
    public static string? LocalsQuery => GetQuery("locals.scm");

    /// <summary>The symbol tagging query for this grammar, or <see langword="null"/> if not embedded.</summary>
    public static string? TagsQuery => GetQuery("tags.scm");

    private static string? GetQuery(string fileName)
    {
        var resourceName = $"TreeSitterIec61131_3St.queries.{fileName}";
        using var stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName);
        if (stream is null) return null;
        using var reader = new System.IO.StreamReader(stream);
        return reader.ReadToEnd();
    }
}

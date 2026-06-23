package tree_sitter_iec61131_3_st_test

import (
	"testing"

	tree_sitter "github.com/tree-sitter/go-tree-sitter"
	tree_sitter_iec61131_3_st "github.com/heytalepazguato/tree-sitter-iec61131-3-st/bindings/go"
)

func TestCanLoadGrammar(t *testing.T) {
	language := tree_sitter.NewLanguage(tree_sitter_iec61131_3_st.Language())
	if language == nil {
		t.Errorf("Error loading Iec611313 grammar")
	}
}

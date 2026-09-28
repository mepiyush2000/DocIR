from docir.reconcile import split_text_to_lines

content = "Teej Tambola: A fun Teej-themed Tambo-\nla was organized, where everyone participated\nenthusiastically and enjoyed the game."
print(split_text_to_lines(content, ["Teej Tambola:Afun Teej-themed Tamo"], [270.0]))

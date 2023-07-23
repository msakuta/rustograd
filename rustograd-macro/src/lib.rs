use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{parse_macro_input, spanned::Spanned, BinOp, Block, Expr, ExprLit, Ident, Pat, Stmt};

#[proc_macro]
pub fn rustograd(input: TokenStream) -> TokenStream {
    // Parse the input tokens into a syntax tree
    let input = parse_macro_input!(input as Block);

    let mut objs = vec![];

    for stmt in &input.stmts {
        traverse_stmt(stmt, &mut objs);
    }

    // Build the output, possibly using quasi-quotation
    let expanded = quote! {
        #(#objs)*
    };

    // Hand the output tokens back to the compiler
    TokenStream::from(expanded)
}

fn traverse_stmt(input: &Stmt, terms: &mut Vec<TokenStream2>) {
    match input {
        Stmt::Local(local) => {
            if let (Pat::Ident(id), Some(init)) = (&local.pat as &Pat, &local.init) {
                let name = id.ident.clone();
                let ex = &init.expr;
                let ts = match ex as &Expr {
                    Expr::Lit(ref lit) => quote! {
                        let #name = ::rustograd::RcTerm::new(stringify!(#name), #lit);
                    },
                    Expr::Path(path) => quote! {
                        let #name = #path;
                    },
                    _ => {
                        if let Some(res) = traverse_expr(ex, terms) {
                            quote! {
                                let #name = #res;
                            }
                        } else {
                            quote! {
                                let #name = ::rustograd::RcTerm::new(stringify!(#name), 1.);
                            }
                        }
                    }
                };
                terms.push(ts);
            }
        }
        Stmt::Expr(ex, _) => {
            traverse_expr(&ex, terms);
        }
        _ => (),
    }
}

fn var_name(terms: &[TokenStream2]) -> String {
    format!("_a{}", terms.len())
}

fn format_term(ex: &ExprLit, terms: &mut Vec<TokenStream2>) -> Ident {
    let name = Ident::new(&var_name(terms), ex.span());
    let ts = quote! {
        let #name = ::rustograd::RcTerm::new(stringify!(#name), #ex);
    };
    terms.push(ts);
    name
}

fn traverse_expr(input: &Expr, terms: &mut Vec<TokenStream2>) -> Option<Ident> {
    match input {
        Expr::Binary(ex) => {
            let lhs = traverse_expr(&ex.left, terms);
            let rhs = traverse_expr(&ex.right, terms);
            if let (Some(lhs), Some(rhs)) = (lhs, rhs) {
                let name = Ident::new(&var_name(terms), ex.span());
                let binop = match ex.op {
                    BinOp::Add(_) => quote! { &#lhs + &#rhs },
                    BinOp::Sub(_) => quote! { &#lhs - &#rhs },
                    BinOp::Mul(_) => quote! { &#lhs * &#rhs },
                    BinOp::Div(_) => quote! { &#lhs / &#rhs },
                    _ => return None,
                };
                let ts = quote! {
                    let #name = #binop;
                };
                terms.push(ts);
                Some(name)
            } else {
                None
            }
        }
        Expr::Paren(ex) => traverse_expr(&ex.expr, terms),
        Expr::Lit(lit) => {
            let name = format_term(&lit, terms);
            Some(name)
        }
        Expr::Path(path) => path.path.segments.last().map(|seg| seg.ident.clone()),
        Expr::Call(call) => {
            if let (Expr::Path(func), Some(arg)) = (&call.func as &Expr, call.args.first()) {
                // let func = path.path.segments.last().map(|seg| seg.ident.clone());
                let name = Ident::new(&var_name(terms), call.span());
                let mut func_derive = func.clone();
                let arg = if let Expr::Path(path) = arg {
                    path.path.segments.last().map(|seg| seg.ident.clone())?
                } else {
                    traverse_expr(arg, terms)?
                };
                if let Some(seg) = func_derive.path.segments.last_mut() {
                    seg.ident = Ident::new(&format!("{}_derive", seg.ident), func.span());
                }
                let ts = quote! {
                    let #name = #arg.apply(stringify!(#func), #func, #func_derive);
                };
                terms.push(ts);
                Some(name)
            } else {
                None
            }
        }
        _ => None,
    }
}

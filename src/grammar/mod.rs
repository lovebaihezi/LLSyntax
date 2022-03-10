mod implement;
mod operator;
pub use implement::*;
pub use operator::*;
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
};

#[derive(Debug)]
pub struct G<T>
where
    T: Hash + Eq + std::fmt::Debug + Symbols + Clone,
{
    start: T,
    action: Vec<Products<T>>,
    map: HashMap<T, usize>,
}

#[derive(Debug, Default, Clone)]
pub struct Product<T>(Vec<T>)
where
    T: Hash + Eq + std::fmt::Debug + Symbols;

unsafe impl<T> Send for Product<T> where T: Hash + Eq + std::fmt::Debug + Symbols + Send {}

#[derive(Debug, Default, Clone)]
pub struct Products<T>(Vec<Product<T>>)
where
    T: Hash + Eq + std::fmt::Debug + Symbols;

unsafe impl<T> Send for Products<T> where T: Hash + Eq + std::fmt::Debug + Symbols + Send {}

#[macro_export]
macro_rules! Symbol {
    {$name:ident, $($id:ident,)+} => {
        #[derive(Debug, Copy, Hash, Clone, PartialEq, Eq)]
        pub enum $name<T>
        where
            T: std::fmt::Debug + Eq + std::hash::Hash + Default,
        {
            $($id,)+
            Terminal(T),
            Epsilon,
            Dollar,
        }
        impl<T> Symbols for $name<T>
        where
            T: std::fmt::Debug + Eq + std::hash::Hash + Default,
        {
            fn is_epsilon(&self) -> bool {
                match self {
                    Self::Epsilon => true,
                    _ => false,
                }
            }
            fn epsilon() -> Self {
                Self::Epsilon
            }
            fn is_nonterminal(&self) -> bool {
                !matches!(self, Self::Terminal(_) | Self::Epsilon | Self::Dollar)
            }
            fn is_dollar(&self) -> bool {
                matches!(self, Self::Dollar)
            }
            fn dollar() -> Self {
                Self::Dollar
            }
        }
        impl<T> Default for $name<T>
        where
            T: std::fmt::Debug + Eq + std::hash::Hash + Default,
        {
            fn default() -> Self {
                Self::Epsilon
            }
        }
        impl<T> std::fmt::Display for $name<T>
        where
            T: std::fmt::Debug + Eq + std::hash::Hash + Default + std::fmt::Display,
        {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::Terminal(v) => f.write_fmt(format_args!("{}", v)),
                    Self::Dollar => f.write_str("$"),
                    Self::Epsilon => f.write_str("𝜀"),
                    v => f.write_fmt(format_args!("{:?}", v))
                }
            }
        }
    };
}

pub trait Symbols {
    fn is_epsilon(&self) -> bool;
    fn epsilon() -> Self;
    fn is_terminal(&self) -> bool {
        !Symbols::is_nonterminal(self) && !Symbols::is_epsilon(self)
    }
    fn is_nonterminal(&self) -> bool;
    fn is_dollar(&self) -> bool;
    fn dollar() -> Self;
}

#[cfg(test)]
mod grammar_test {
    #[test]
    fn build_grammar() {}
}

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
    cache: Vec<[HashSet<T>; 2]>,
    action: Vec<Products<T>>,
    map: HashMap<T, usize>,
}

#[derive(Debug, Default, Clone)]
pub struct Product<T>(Vec<T>)
where
    T: Hash + Eq + std::fmt::Debug + Symbols;

#[derive(Debug, Default, Clone)]
pub struct Products<T>(Vec<Product<T>>)
where
    T: Hash + Eq + std::fmt::Debug + Symbols;

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
                !matches!(self, Self::Terminal(_) | Self::Epsilon)
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
    };
}

// pub trait LLGrammar {
//     fn is_left_recursion(&self) -> bool;
//     fn is_cycle(&self) -> bool;
//     fn is_epilison_product(&self) -> bool;
//     type EliminationLeftRecursion;
//     fn elimination_left_recursion(&self) -> Self::EliminationLeftRecursion;
//     // fn elimination_cycle<T: LLGrammar>(&self) -> T;
//     // fn elimination_epilison(&self) -> LLGrammar;
//     fn follow<V: Symbols, T: IntoIterator>(&self, sign: V) -> T;
//     fn first<V: Symbols, T: IntoIterator>(&self, sign: V) -> T;
//     // fn left_factoring(&self) -> impl LL;
// }

pub trait Symbols {
    fn is_epsilon(&self) -> bool;
    fn epsilon() -> Self;
    fn is_terminal(&self) -> bool {
        !self.is_nonterminal()
    }
    fn is_nonterminal(&self) -> bool;
}

#[cfg(test)]
mod grammar_test {
    #[test]
    fn build_grammar() {}
}

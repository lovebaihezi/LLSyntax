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
    action: Vec<Construct<T>>,
    map: HashMap<T, usize>,
}

#[derive(Debug, Default, Clone)]
pub struct Action<T>(Vec<T>)
where
    T: Hash + Eq + std::fmt::Debug + Symbols;

#[derive(Debug, Default, Clone)]
pub struct Actions<T>(Vec<Action<T>>)
where
    T: Hash + Eq + std::fmt::Debug + Symbols;

#[derive(Debug, Default, Clone)]
pub struct Construct<T>(Actions<T>)
where
    T: Hash + Eq + std::fmt::Debug + Symbols + Clone;

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
            Epilison,
        }
        impl<T> Symbols for $name<T>
        where
            T: std::fmt::Debug + Eq + std::hash::Hash + Default,
        {
            fn is_epilison(&self) -> bool {
                match self {
                    Self::Epilison => true,
                    _ => false,
                }
            }
            fn epilison() -> Self {
                Self::Epilison
            }
            fn is_nonterminal(&self) -> bool {
                !matches!(self, Self::Terminal(_) | Self::Epilison)
            }
        }
        impl<T> Default for $name<T>
        where
            T: std::fmt::Debug + Eq + std::hash::Hash + Default,
        {
            fn default() -> Self {
                Self::Epilison
            }
        }
    };
}

pub trait Symbols {
    fn is_epilison(&self) -> bool;
    fn epilison() -> Self;
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

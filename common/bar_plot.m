clear; close all; clc;


X = categorical({'Morningglory', 'Carpetweed', 'Palmer Amaranth', 'Waterhemp', 'Purslane', 'Nutsedge', 'Eclipta', 'Sicklepod', 'Spotted Spurge', 'Goosegrass', 'Prickly Sida', 'Ragweed', 'Crabgrass', 'Swinecress', 'Spurred Anoda'});
X = reordercats(X,{'Morningglory', 'Carpetweed', 'Palmer Amaranth', 'Waterhemp', 'Purslane', 'Nutsedge', 'Eclipta', 'Sicklepod', 'Spotted Spurge', 'Goosegrass', 'Prickly Sida', 'Ragweed', 'Crabgrass', 'Swinecress', 'Spurred Anoda'});

nums = [1115 763 689 451 450 273 254 240 234 216 129 129 111 72 61];

y = [724 223 168;
    495 153 115;
    447 138 104;
    292 91 68;
    292 90 68;
    177 55 41;
    164 51 39;
    156 48 36;
    151 47 36;
    139 44 33;
    83 26 20;
    83 26 20;
    71 23 17;
    46 15 11;
    38 13 10];

h = bar(X, y,'stacked', 'BarWidth', 0.7)
for i1=1:15
    text(X(i1),nums(i1),num2str(nums(i1)),...
               'HorizontalAlignment','center',...
               'VerticalAlignment','bottom')
end

set(h, {'DisplayName'}, {'train','val','test'}')
legend()
ylabel('Number of images', 'FontSize', 12)

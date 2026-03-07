const {getDefaultConfig, mergeConfig} = require('@react-native/metro-config');

const defaultConfig = getDefaultConfig(__dirname);

module.exports = mergeConfig(defaultConfig, {
  projectRoot: __dirname,
  watchFolders: [__dirname],
  resolver: {
    sourceExts: [...defaultConfig.resolver.sourceExts, 'cjs'],
  },
});
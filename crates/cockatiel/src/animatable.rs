use bevy::{
  ecs::component::{Component, Mutable},
  image::TextureAtlas,
  sprite::Sprite,
  ui::widget::ImageNode,
};

pub trait Animatable: Component<Mutability = Mutable> {
  fn get_texture_atlas_mut(&mut self) -> &mut Option<TextureAtlas>;
  fn set_flip(&mut self, flip_x: bool, flip_y: bool);
}

impl Animatable for ImageNode {
  fn get_texture_atlas_mut(&mut self) -> &mut Option<TextureAtlas> {
    &mut self.texture_atlas
  }

  fn set_flip(&mut self, flip_x: bool, flip_y: bool) {
    self.flip_x = flip_x;
    self.flip_y = flip_y;
  }
}

impl Animatable for Sprite {
  fn get_texture_atlas_mut(&mut self) -> &mut Option<TextureAtlas> {
    &mut self.texture_atlas
  }

  fn set_flip(&mut self, flip_x: bool, flip_y: bool) {
    self.flip_x = flip_x;
    self.flip_y = flip_y;
  }
}
